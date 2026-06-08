"""Sequence multi-pressure model inference and prediction aggregation.

Sample run with repo ``sample_data`` (after sequence training writes a checkpoint)::

  python predict_gas_ads_seq.py
    --model-path /path/to/model.pt
    --grid-path ./sample_data
    --load-dir ./sample_data/all_seq
    --target Kr
    --arch seq
    --seq-order forward
    --seed 168
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics.functional import r2_score

from model_property_all import (
    ResNet,
    ResNetMultiOutput,
    ResNetSeq,
    SeqLearningMethod,
    SeqDataset,
)

TARGET_COLUMNS = {"Xe": "Xe_cm3_per_cm3_value", "Kr": "Kr_cm3_per_cm3_value"}

PRESSURE_TAGS = ["0p1bar", "0p25bar", "0p5bar", "1bar", "5bar", "10bar"]
PRESSURE_FEATURES = ["Kr_Fugacity", "Xe_Fugacity"]

DEFAULT_GRID = "./sample_data"
OUTPUT_DIR = "predictions"


def parse_args():
    p = argparse.ArgumentParser(description="Sequence gas-adsorption inference (checkpoint required).")
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Checkpoint (*.pt) with model_state_dict.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--target",
        choices=list(TARGET_COLUMNS.keys()),
        default="Kr",
        help="Must match CSV / training column",
    )
    p.add_argument("--load-dir", type=str, default="./grids", help="all_3grids CSV folder")
    p.add_argument("--grid-path", type=str, default=DEFAULT_GRID, help="train/ val/ test/")
    p.add_argument(
        "--arch",
        choices=["multioutput", "rnn"],
        default="multioutput",
        help="Must match checkpoint architecture.",
    )
    p.add_argument(
        "--seq-order",
        choices=["forward", "reverse"],
        default="forward",
    )
    p.add_argument(
        "--softplus",
        action="store_true",
        help="ResNetSeq only: match trained softplus flag.",
    )
    p.add_argument("--num-grids", type=int, default=3)
    p.add_argument("--grid-type", type=str, default="")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip derive_pred_multi_grid_seq (use when num_grids==1)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_seq_data_id(
    dir_batch: str,
    path_to_csv: str,
    target_name: str,
    index_col: str,
    pressure_cols: list,
    *,
    reverse: bool,
    size=None,
):
    with open(os.path.join(dir_batch, "clean.json")) as fh:
        names = json.load(fh)["name"]
    print(dir_batch, len(names))

    df = pd.read_csv(path_to_csv)
    df = df[df[index_col].isin(names)]
    df = df.sort_values(by=[index_col, pressure_cols[0]], ascending=[True, not reverse])
    grouped = df.groupby(index_col)

    y_df = grouped[target_name].apply(lambda x: x.values).reindex(names)
    y = np.stack(y_df.values)

    p_df = grouped[pressure_cols].apply(lambda x: x.values).reindex(names)
    p_feat = np.stack(p_df.values)

    X = np.load(os.path.join(dir_batch, "clean.npy"), mmap_mode="r")
    if size is not None:
        X = X[:size]
        y = y[:size]
        p_feat = p_feat[:size]
        names = names[:size]

    return X, y, p_feat, names


def get_extras_list(df_all: pd.DataFrame, batch_dir: str, index_col: str):
    with open(os.path.join(batch_dir, "clean.json")) as f:
        names = json.load(f)["name"]
    df_g = df_all.groupby(index_col).first()
    return df_g.loc[names, ["PLD", "LCD"]].values.astype(np.float32)


def build_network(
    arch: str,
    grid_type: str,
    *,
    pressure_dim: int | None,
    softplus: bool,
    device,
):
    ch = 3 if grid_type == "concat" else 1
    property_cols_dim = len(["PLD", "LCD"])
    if arch == "multioutput":
        net = ResNetMultiOutput(
            extra_dim=2,
            channel=ch,
            num_outputs=6,
        ).to(device)
    else:
        base = ResNet(extra_dim=property_cols_dim, channel=ch)
        net = ResNetSeq(
            base_cnn=base,
            pressure_dim=pressure_dim or 2,
            rnn_hidden_dim=32,
            rnn_type="GRU",
            bidirection=False,
            num_layers=1,
            softplus=softplus,
        ).to(device)
    return net


def run_inference(args, target_col: str):
    reverse = args.seq_order == "reverse"
    index_col = "sample"
    grid_type = args.grid_type
    load_dir = args.load_dir
    grid_root = args.grid_path
    test_dir = os.path.join(grid_root, "test")

    csv_path = os.path.join(
        load_dir, f"all_{args.num_grids}grids_{grid_type}".rstrip("_") + ".csv"
    )
    print(csv_path)

    df_all = pd.read_csv(csv_path, index_col=index_col)

    X_te, y_te, p_te, ids_test = load_seq_data_id(
        test_dir,
        csv_path,
        target_col,
        index_col,
        PRESSURE_FEATURES,
        reverse=reverse,
    )

    assert np.isnan(X_te).sum() == 0 and np.isnan(y_te).sum() == 0

    extras_test = get_extras_list(df_all, test_dir, index_col)

    if grid_type != "concat":
        X_te = X_te.reshape(-1, 1, X_te.shape[1], X_te.shape[2], X_te.shape[3])
    else:
        X_te = np.transpose(X_te, (0, 4, 1, 2, 3))

    std = transforms.Normalize(
        mean=X_te.mean(),
        std=max(float(X_te.std()), 1e-6),
    )
    ds = SeqDataset(
        X=X_te,
        y=y_te,
        extras=extras_test,
        pressures=p_te if args.arch == "rnn" else None,
        transform_X=std,
        transform_y=None,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdim = p_te.shape[2] if p_te.ndim == 3 else None
    net = build_network(args.arch, grid_type, pressure_dim=pdim, softplus=args.softplus, device=device)

    ckpt_path = Path(args.model_path)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    model = SeqLearningMethod(net, None, None, logger=None)
    preds = []
    truths = []

    use_pressure = args.arch == "rnn"

    with torch.no_grad():
        if use_pressure:
            for xb, yb, ex, pb in loader:
                xb = xb.to(device)
                ex = ex.to(device)
                pb = pb.to(device)
                y_hat = model.predict(xb, ex, pb).cpu()
                preds.append(y_hat)
                truths.append(yb.numpy())
        else:
            for xb, yb, ex in loader:
                xb = xb.to(device)
                ex = ex.to(device)
                y_hat = model.predict(xb, ex, None).cpu()
                preds.append(y_hat)
                truths.append(yb.numpy())

    y_pred = torch.cat(preds, dim=0).numpy()
    y_true = np.concatenate(truths, axis=0)

    N, T = y_true.shape
    if T != len(PRESSURE_TAGS):
        raise ValueError(f"Expected T={len(PRESSURE_TAGS)} outputs, got {T}")

    y_flat_p = y_pred.reshape(-1)
    y_flat_t = y_true.reshape(-1)

    rmse_flat = math.sqrt(sk_mse(y_flat_t, y_flat_p))
    print(
        f"Overall torcheval R2 {r2_score(torch.tensor(y_flat_p), torch.tensor(y_flat_t)).item():.4f}",
        f" sklearn R2 {sk_r2(y_flat_t, y_flat_p):.4f} RMSE {rmse_flat:.4f} MAE {sk_mae(y_flat_t, y_flat_p):.4f}",
    )

    tag_order = PRESSURE_TAGS[::-1] if reverse else PRESSURE_TAGS
    slug_list = tag_order

    cols = {}
    cols["sample"] = ids_test
    for j, tag in enumerate(slug_list):
        cols[f"true_{tag}"] = y_true[:, j]
        cols[f"pred_{tag}"] = y_pred[:, j]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem
    pred_csv = out_dir / f"pred_{stem}.csv"
    pd.DataFrame(cols).to_csv(pred_csv, index=False)
    print(f"Wrote {pred_csv}")

    return pd.DataFrame(cols), stem


def derive_pred_multi_grid_seq(
    df_pred: pd.DataFrame,
    pressure_tags: list,
    sample_col: str,
    out_dir: Path,
    stem: str,
):
    need = [sample_col] + [f"true_{p}" for p in pressure_tags] + [f"pred_{p}" for p in pressure_tags]
    missing = [c for c in need if c not in df_pred.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df_pred.copy()
    df["mof"] = df[sample_col].astype(str).str.split("--").str[0]
    df["grid_type"] = df[sample_col].astype(str).str.split("--").str[1]

    agg_spec = {f"true_{p}": "first" for p in pressure_tags}
    agg_spec.update({f"pred_{p}": "median" for p in pressure_tags})
    grp = df.groupby("mof").agg(agg_spec).reset_index()

    agg_path = out_dir / f"{stem}_agg_mof.csv"
    grp.to_csv(agg_path, index=False)
    print(f"Wrote aggregated {agg_path}")

    for j, pt in enumerate(pressure_tags):
        yt = grp[f"true_{pt}"].values
        yp = grp[f"pred_{pt}"].values
        r2 = sk_r2(yt, yp)
        print(f"Agg [{pt}] R2={r2:.4f} MAE={sk_mae(yt, yp):.4f} RMSE={math.sqrt(sk_mse(yt, yp)):.4f}")

    return grp


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    target_col = TARGET_COLUMNS[args.target]
    df_pred, ckpt_stem = run_inference(args, target_col)

    if args.num_grids > 1 and not args.no_aggregate:
        derive_pred_multi_grid_seq(
            df_pred,
            PRESSURE_TAGS,
            "sample",
            Path(args.output_dir),
            ckpt_stem,
        )


if __name__ == "__main__":
    main()
