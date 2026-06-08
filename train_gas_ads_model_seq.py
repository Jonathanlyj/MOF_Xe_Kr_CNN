"""Sample run::

  python train_gas_ads_model_seq.py 
    --grid-path ./sample_data 
    --load-dir ./sample_data/all_seq 
    --model-save-dir ./multi_pressure_models 
    --output-dir ./multi_pressure_models 
    --target Kr 
    --seed 168
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics.functional import r2_score

from model_property_all import (
    ResNet,
    ResNetSeq,
    ResNetMultiOutput,
    SeqLearningMethod,
    SeqDataset,
    init_weights,
    ThresholdRelativeWeightedMAELoss,
    AsymmetricMAELoss,
    AdaptiveAsymmetricMAELoss,
)

TARGET_COLUMNS = {"Xe": "Xe_cm3_per_cm3_value", "Kr": "Kr_cm3_per_cm3_value"}

PRESSURE_SEQUENCE = ["0.1bar", "0.25bar", "0.5bar", "1bar", "5bar", "10bar"]

PRESSURE_SLUG_MAP = {
    "0.1bar": "0p1bar",
    "1bar": "1bar",
    "5bar": "5bar",
    "10bar": "10bar",
    "0.25bar": "0p25bar",
    "0.5bar": "0p5bar",
    "all_seq": "all_seq",
    "all_seq_reverse": "all_seq_reverse",
}

HYPER = {
    "batch_size": 64,
    "num_epochs": 1, # set to 250 for full training
    "eval_freq": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "step_size": 60,
    "gamma": 0.5,
    "optimizer": "Adam",
    "patience": 100,
    "rnn_num_layers": 1,
    "rnn_bidirectional": False,
    "rnn_hidden_dim": 32,
    "rnn_type": "GRU",
    "softplus": False,
    "loss_type": None,
}

SEQ_MODEL_TYPE = "multioutput"
PRESSURE_FEATURE_COLS = ["Kr_Fugacity", "Xe_Fugacity"]


def parse_args():
    p = argparse.ArgumentParser(description="Train sequence gas-adsorption model (ResNet + seq heads).")
    p.add_argument(
        "--target",
        choices=list(TARGET_COLUMNS.keys()),
        default="Kr",
        help="Xe or Kr (CSV column)",
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument(
        "--load-dir",
        default="./grids",
        help="Folder with all_<N>grids CSV (sequence labels / fugacity)",
    )
    p.add_argument(
        "--grid-path",
        default="./sample_data",
        help="train/ val/ test/ batches",
    )
    p.add_argument(
        "--model-save-dir",
        default="./models",
        help="Checkpoints (*.pt)",
    )
    p.add_argument("--output-dir", default="pred", help="Prediction and metrics CSV")
    p.add_argument(
        "--seq-order",
        choices=["forward", "reverse"],
        default="forward",
        help="forward=all_seq pressures; reverse=all_seq_reverse sorting",
    )
    p.add_argument(
        "--under-weight",
        type=float,
        default=None,
        help="Asymmetric MAELoss under-weight (only used if loss-type is set to asymm in code).",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logger(log_dir: str, log_filename: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_filename)),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_seq_data_id(
    dir_batch,
    path_to_csv,
    target_name,
    index_col,
    pressure_feature_cols,
    *,
    reverse: bool,
    size=None,
):
    with open(os.path.join(dir_batch, "clean.json"), "r") as fh:
        names = json.load(fh)["name"]
    print(dir_batch, len(names))

    df = pd.read_csv(path_to_csv)
    df = df[df[index_col].isin(names)]
    df = df.sort_values(by=[index_col, pressure_feature_cols[0]], ascending=[True, not reverse])
    grouped = df.groupby(index_col)

    y_df = grouped[target_name].apply(lambda x: x.values).reindex(names)
    y = np.stack(y_df.values)
    p_df = grouped[pressure_feature_cols].apply(lambda x: x.values).reindex(names)
    p = np.stack(p_df.values)

    X = np.load(os.path.join(dir_batch, "clean.npy"), mmap_mode="r")

    if size is not None:
        X = X[:size]
        y = y[:size]
        p = p[:size]
        names = names[:size]

    return X, y, p, names


def get_extras_list(df_all: pd.DataFrame, dir_path: str, index_col: str):
    with open(os.path.join(dir_path, "clean.json"), "r") as f:
        names = json.load(f)["name"]
    df_grouped = df_all.groupby(index_col).first()
    return df_grouped.loc[names, ["PLD", "LCD"]].values.astype(np.float32)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = parse_args()

    pressure_key = "all_seq_reverse" if args.seq_order == "reverse" else "all_seq"
    pressure_slug = PRESSURE_SLUG_MAP[pressure_key]
    target_col = TARGET_COLUMNS[args.target]
    hyper = {**HYPER, "under_weight": args.under_weight}
    reverse = args.seq_order == "reverse"

    set_seed(args.seed)

    grid_dir = args.grid_path
    load_dir = args.load_dir
    index_col = "sample"
    grid_type = ""
    num_grids = 3
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    csv_path_data = os.path.join(
        load_dir, f"all_{num_grids}grids_{grid_type}".strip("_") + ".csv"
    )
    train_dir = os.path.join(grid_dir, "train")
    val_dir = os.path.join(grid_dir, "val")
    test_dir = os.path.join(grid_dir, "test")

    model_name_core = (
        f"MixNewSplit{pressure_slug}_{num_grids}_grids_{grid_type}_{target_col}_no_64_{timestamp}"
        f"_seed{args.seed}"
    )
    uw_part = (
        f"uw_{hyper['under_weight']}_"
        if hyper["loss_type"] == "asymm"
        else ""
    )
    model_name = (
        f"{model_name_core}_"
        f"loss_{hyper['loss_type']}_"
        f"{uw_part}"
        f"{hyper['batch_size']}_"
        f"{hyper['num_epochs']}_"
        f"{hyper['learning_rate']}_"
        f"{hyper['weight_decay']}_"
        f"{hyper['step_size']}_"
        f"{hyper['gamma']}_"
        f"{SEQ_MODEL_TYPE}_"
        f"{hyper['optimizer']}_"
        f"{hyper['rnn_num_layers']}_"
        f"{hyper['rnn_bidirectional']}_"
        f"{hyper['rnn_hidden_dim']}_"
        f"{hyper['rnn_type']}_"
        f"{hyper['softplus']}"
    )

    logger = setup_logger("./log", f"{model_name}_{timestamp}.log")
    logger.info("Target: %s  seq_order=%s", target_col, args.seq_order)
    logger.info("Hyperparameters: %s", hyper)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X_train, y_train, p_train, _ = load_seq_data_id(
        train_dir,
        csv_path_data,
        target_col,
        index_col,
        PRESSURE_FEATURE_COLS,
        reverse=reverse,
    )
    X_val, y_val, p_val, _ = load_seq_data_id(
        val_dir,
        csv_path_data,
        target_col,
        index_col,
        PRESSURE_FEATURE_COLS,
        reverse=reverse,
    )
    X_test, y_test, p_test, ids_test = load_seq_data_id(
        test_dir,
        csv_path_data,
        target_col,
        index_col,
        PRESSURE_FEATURE_COLS,
        reverse=reverse,
    )

    logger.info(
        "X_train %s p_train %s y_train %s",
        X_train.shape,
        p_train.shape,
        y_train.shape,
    )

    df_all = pd.read_csv(csv_path_data, index_col=index_col)
    extras_train = get_extras_list(df_all, train_dir, index_col)
    extras_val = get_extras_list(df_all, val_dir, index_col)
    extras_test = get_extras_list(df_all, test_dir, index_col)

    assert not np.isnan(extras_train).any()
    assert not np.isnan(extras_val).any()
    assert not np.isnan(extras_test).any()

    property_cols = ["PLD", "LCD"]

    if grid_type != "concat":
        X_train = X_train.reshape(-1, 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
        X_val = X_val.reshape(-1, 1, X_val.shape[1], X_val.shape[2], X_val.shape[3])
        X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    else:
        X_train = np.transpose(X_train, (0, 4, 1, 2, 3))
        X_val = np.transpose(X_val, (0, 4, 1, 2, 3))
        X_test = np.transpose(X_test, (0, 4, 1, 2, 3))

    tr_std = max(float(X_train.std()), 1e-6)
    va_std = max(float(X_val.std()), 1e-6)
    te_std = max(float(X_test.std()), 1e-6)
    std_train = transforms.Normalize(mean=X_train.mean(), std=tr_std)
    std_val = transforms.Normalize(mean=X_val.mean(), std=va_std)
    std_test = transforms.Normalize(mean=X_test.mean(), std=te_std)

    y_normalizer = None
    train_ds = SeqDataset(
        X=X_train,
        y=y_train,
        extras=extras_train,
        pressures=p_train if SEQ_MODEL_TYPE == "rnn" else None,
        transform_X=std_train,
        transform_y=y_normalizer,
    )
    val_ds = SeqDataset(
        X=X_val,
        y=y_val,
        extras=extras_val,
        pressures=p_val if SEQ_MODEL_TYPE == "rnn" else None,
        transform_X=std_val,
        transform_y=y_normalizer,
    )
    test_ds = SeqDataset(
        X=X_test,
        y=y_test,
        extras=extras_test,
        pressures=p_test if SEQ_MODEL_TYPE == "rnn" else None,
        transform_X=std_test,
        transform_y=y_normalizer,
    )

    bs_train = hyper["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        pin_memory=True,
        drop_last=len(train_ds) >= bs_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2 * hyper["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=2 * hyper["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    pressure_dim = p_train.shape[2]
    if SEQ_MODEL_TYPE == "rnn":
        base = ResNet(
            extra_dim=len(property_cols),
            channel=3 if grid_type == "concat" else 1,
        )
        net = ResNetSeq(
            base_cnn=base,
            pressure_dim=pressure_dim,
            rnn_hidden_dim=hyper["rnn_hidden_dim"],
            rnn_type=hyper["rnn_type"],
            bidirection=hyper["rnn_bidirectional"],
            num_layers=hyper["rnn_num_layers"],
            softplus=hyper["softplus"],
        ).to(device)
    else:
        net = ResNetMultiOutput(
            extra_dim=2,
            channel=3 if grid_type == "concat" else 1,
            num_outputs=6,
        ).to(device)

    total_p, train_p = count_parameters(net)
    logger.info("Parameters total %s trainable %s", f"{total_p:,}", f"{train_p:,}")

    if hyper["loss_type"] == "threshold":
        criterion = ThresholdRelativeWeightedMAELoss(
            threshold=5.0, eps=1e-2, sharpness=2.0
        ).to(device)
    elif hyper["loss_type"] == "asymm":
        criterion = AsymmetricMAELoss(
            under_weight=hyper["under_weight"],
        ).to(device)
    elif hyper["loss_type"] == "adap_asymm":
        criterion = AdaptiveAsymmetricMAELoss(eps=1e-2).to(device)
    else:
        criterion = nn.L1Loss().to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyper["step_size"],
        gamma=hyper["gamma"],
        verbose=True,
    )

    net.apply(lambda m: init_weights(m, a=0.01))
    if SEQ_MODEL_TYPE == "rnn":
        with torch.no_grad():
            net.output_head[-1].bias.fill_(y_train.mean())
    else:
        with torch.no_grad():
            net.fc2[-1].bias.fill_(y_train.mean())
    logger.info("Init final bias ~ %.4f", float(y_train.mean()))

    model = SeqLearningMethod(
        network=net,
        optimizer=optimizer,
        criterion=criterion,
        logger=logger,
    )
    model.net.to(device)

    model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        val_loss_freq=hyper["eval_freq"],
        epochs=hyper["num_epochs"],
        scheduler=scheduler,
        metric=r2_score,
        device=device,
        verbose=True,
        patience=hyper["patience"],
        pressure_input=SEQ_MODEL_TYPE == "rnn",
    )

    model.net.eval()
    predictions = []
    true_values = []
    for batch in test_loader:
        if SEQ_MODEL_TYPE == "rnn":
            xb, yb, extras_b, pressures_b = batch
        else:
            xb, yb, extras_b = batch
            pressures_b = None
        xb = xb.to(device)
        extras_b = extras_b.to(device)
        if SEQ_MODEL_TYPE == "rnn":
            pressures_b = pressures_b.to(device)

        with torch.no_grad():
            if SEQ_MODEL_TYPE == "rnn":
                y_pb = model.predict(xb, extras_b, pressures_b)
            else:
                y_pb = model.predict(xb, extras_b)
        predictions.append(y_pb.cpu())
        true_values.append(yb)

    y_pred = torch.vstack(predictions).numpy()
    y_true = torch.vstack(true_values).numpy()

    if y_normalizer is not None:
        y_pred = y_normalizer.inverse(torch.tensor(y_pred)).numpy()
        y_true = y_normalizer.inverse(torch.tensor(y_true)).numpy()

    T = y_true.shape[1]
    metrics = {"R2": [], "MSE": [], "RMSE": [], "MAE": [], "%MAE": []}
    logger.info("Per timestep metrics")
    for t in range(T):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        r2_t = sk_r2(yt, yp)
        mse_t = sk_mse(yt, yp)
        rmse_t = np.sqrt(mse_t)
        mae_t = sk_mae(yt, yp)
        mtr = np.mean(yt)
        pct = (mae_t / mtr) * 100 if mtr != 0 else float("inf")
        metrics["R2"].append(r2_t)
        metrics["MSE"].append(mse_t)
        metrics["RMSE"].append(rmse_t)
        metrics["MAE"].append(mae_t)
        metrics["%MAE"].append(pct)
        logger.info(
            "Pressure %s | R2 %.3f | MSE %.4f | RMSE %.4f | MAE %.4f | %%MAE %.2f%%",
            PRESSURE_SEQUENCE[t],
            r2_t,
            mse_t,
            rmse_t,
            mae_t,
            pct,
        )

    df_metrics = pd.DataFrame(metrics, index=range(T)).T
    df_metrics.columns = list(PRESSURE_SEQUENCE)

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    r2_all = sk_r2(y_true_f, y_pred_f)
    mse_all = sk_mse(y_true_f, y_pred_f)
    rmse_all = np.sqrt(mse_all)
    mae_all = sk_mae(y_true_f, y_pred_f)
    m_all = np.mean(y_true_f)
    pct_all = (mae_all / m_all) * 100 if m_all != 0 else float("inf")
    logger.info(
        "Overall R2 %.3f MSE %.4f RMSE %.4f MAE %.4f %%MAE %.2f%%",
        r2_all,
        mse_all,
        rmse_all,
        mae_all,
        pct_all,
    )
    df_metrics["Overall"] = [r2_all, mse_all, rmse_all, mae_all, pct_all]

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_csv = os.path.join(args.output_dir, f"{model_name}_seq_pred_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=True)
    logger.info("Metrics saved %s", metrics_csv)

    os.makedirs(args.model_save_dir, exist_ok=True)
    ckpt = os.path.join(args.model_save_dir, f"{model_name}_state_dict.pt")
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": hyper["num_epochs"],
        },
        ckpt,
    )
    logger.info("Checkpoint %s", ckpt)

    slug_list = (
        list(PRESSURE_SEQUENCE[::-1]) if reverse else list(PRESSURE_SEQUENCE)
    )
    df_pred_t = pd.DataFrame(
        y_true,
        columns=[f"true_{PRESSURE_SLUG_MAP[p]}" for p in slug_list],
    )
    df_pred_p = pd.DataFrame(
        y_pred,
        columns=[f"pred_{PRESSURE_SLUG_MAP[p]}" for p in slug_list],
    )
    df_predictions = pd.concat([df_pred_t, df_pred_p], axis=1)
    df_predictions.index = ids_test
    df_predictions.index.name = "sample_id"
    pred_path = os.path.join(args.output_dir, f"pred_{model_name}.csv")
    df_predictions.to_csv(pred_path, index=True)
    logger.info("Predictions saved %s", pred_path)


if __name__ == "__main__":
    main()
