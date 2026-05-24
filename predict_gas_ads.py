"""Inference for Xe/Kr adsorption CNN.

Expect:
  --grid-path/test/: clean.json, clean.npy (voxel tensors and sample order)
  --load-dir/all_<N>grids*.csv labels (N=3, empty grid suffix)


Sample run with repo ``sample_data`` (after training writes a checkpoint)::

  python predict_gas_ads.py \
    --model-path /path/to/model.pt \
    --grid-path ./sample_data \
    --load-dir ./sample_data/1bar \
    --target Xe \
    --seed 42
"""

import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics.functional import r2_score

from model_property_all import CustomDataset, LearningMethod, RetNet, load_data_id
from model_property_all_swin import RetNetSwin

TARGET_COLUMNS = {"Xe": "Xe_cm3_per_cm3_value", "Kr": "Kr_cm3_per_cm3_value"}
OUTPUT_DIR = "predictions"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _infer_model_type(model_path: str) -> str:
    base = os.path.basename(model_path).lower()
    return "swin" if "swin" in base else "retnet"


def get_extras_list(
    load_dir: str,
    grid_path: str,
    num_grids: int,
    grid_type: str,
):
    csv_path = os.path.join(
        load_dir, f"all_{num_grids}grids_{grid_type}".rstrip("_") + ".csv"
    )
    json_path = os.path.join(grid_path, "test", "clean.json")
    df_all = pd.read_csv(csv_path, index_col="sample")
    with open(json_path, "r") as f:
        info = json.load(f)
    names = info["name"]
    return df_all.loc[names, ["PLD", "LCD"]].values.astype(np.float32)


def model_predict(
    model_path: str,
    target_col: str,
    load_dir: str,
    grid_path: str,
    num_grids: int = 3,
    grid_type: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = _infer_model_type(model_path)
    ch = 3 if grid_type == "concat" else 1

    if model_type == "retnet":
        net = RetNet(extra_dim=2, channel=ch).to(device)
    else:
        net = RetNetSwin(extra_dim=2, channel=ch).to(device)

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)

    test_batch_dir = os.path.join(grid_path, "test")
    csv_path = os.path.join(
        load_dir, f"all_{num_grids}grids_{grid_type}".rstrip("_") + ".csv"
    )
    print(csv_path)
    print(test_batch_dir)

    X_test, y_test, ids_test = load_data_id(
        test_batch_dir,
        csv_path,
        target_col,
        "sample",
    )
    extras_test = get_extras_list(load_dir, grid_path, num_grids=num_grids, grid_type=grid_type)

    assert np.isnan(X_test).sum() == 0
    assert np.isnan(y_test).sum() == 0

    standardization = transforms.Normalize(
        X_test.mean(), max(float(X_test.std()), 1e-6)
    )

    if grid_type != "concat":
        X_test = X_test.reshape(
            -1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3]
        )
    else:
        X_test = np.transpose(X_test, (0, 4, 1, 2, 3))

    print(X_test.shape, y_test.shape)

    test_dataset = CustomDataset(
        X=X_test, y=y_test, extras=extras_test, transform_X=standardization
    )
    test_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True)

    model = LearningMethod(net, None, None, logger=None)
    predictions = []
    for i, (xb, _, extras_b) in enumerate(test_loader):
        print(f"Batch {i + 1}/{len(test_loader)}")
        xb, extras_b = xb.to(device), extras_b.to(device)
        with torch.no_grad():
            y_pred_b = model.predict(xb, extras_b)
        predictions.append(y_pred_b.cpu())

    y_pred = torch.cat(predictions).numpy()
    y_true = y_test.reshape(len(y_test), -1)

    r2 = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)


    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    id_col = "sample"
    df_test = pd.DataFrame(
        {
            id_col: ids_test,
            target_col: y_true,
            f"{target_col}_pred": y_pred,
        }
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(model_path))[0]
    csv_path_out = os.path.join(OUTPUT_DIR, f"{stem}_pred.csv")
    df_test.to_csv(csv_path_out, index=False)
    print(f"Wrote {csv_path_out}")
    return df_test


def derive_pred_multi_grid(
    df_pred: pd.DataFrame,
    col: str,
    out_stem: str,
):
    df_pred = df_pred.copy()
    df_pred["mof"] = df_pred["sample"].apply(lambda x: x.split("--")[0])
    df_pred["grid_type"] = df_pred["sample"].apply(lambda x: x.split("--")[1])
    df_pred_mof = (
        df_pred.groupby("mof")
        .agg(y_true=(f"{col}", "first"), y_pred=(f"{col}_pred", "median"))
        .reset_index()
    )
    df_pred_mof[col] = df_pred_mof["y_true"]
    df_pred_mof[f"{col}_pred"] = df_pred_mof["y_pred"]
    y_pred = df_pred_mof["y_pred"].values
    y_true = df_pred_mof["y_true"].values

    r2 = r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    print(f"Agg pred R^2: {r2:.3f}")
    print(f"Agg pred MAE: {mae:.3f}")
    print(f"Agg pred RMSE: {rmse:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    agg_csv = os.path.join(OUTPUT_DIR, f"grid_agg_{out_stem}_pred.csv")
    df_pred_mof.to_csv(agg_csv, index=False)
    print(f"Wrote {agg_csv}")
    return df_pred_mof


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained model on test grids (Xe/Kr adsorption)."
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed (default: 1).")
    p.add_argument(
        "--target",
        choices=["Xe", "Kr"],
        default="Xe",
        help="Xe -> Xe_cm3_per_cm3_value; Kr -> Kr_cm3_per_cm3_value.",
    )
    p.add_argument(
        "--load-dir",
        type=str,
        default="./grids",
        help="Directory with all_<N>grids CSV (targets / metadata)",
    )
    p.add_argument(
        "--grid-path",
        type=str,
        default="./sample_data",
        help="Batch root with test/clean.json and test/clean.npy",
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved checkpoint (.pt).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    target_col = TARGET_COLUMNS[args.target]
    num_grids = 3
    grid_type = ""

    df_pred = model_predict(
        args.model_path,
        target_col,
        args.load_dir,
        grid_path=args.grid_path,
        num_grids=num_grids,
        grid_type=grid_type,
    )
    stem = os.path.splitext(os.path.basename(args.model_path))[0]

    if num_grids > 1 and grid_type != "concat":
        derive_pred_multi_grid(df_pred, target_col, stem)


if __name__ == "__main__":
    main()