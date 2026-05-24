"""Sample run::

  python train_gas_ads_model.py \
    --pressure 1bar \
    --grid-path ./sample_data \
    --load-dir ./sample_data/1bar \
    --model-save-dir ./single_pressure_models \
    --output-dir ./single_pressure_models \
    --target Xe \
    --seed 42
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from torchvision import transforms
from torcheval.metrics.functional import r2_score

from model_property_all import (
    RetNet,
    LearningMethod,
    CustomDataset,
    init_weights,
    load_data,
)

TARGET_COLUMNS = {
    "Xe": "Xe_cm3_per_cm3_value",
    "Kr": "Kr_cm3_per_cm3_value",
}

PRESSURE_SLUG = {
    "0.1bar": "0p1bar",
    "1bar": "1bar",
    "5bar": "5bar",
    "10bar": "10bar",
    "0.25bar": "0p25bar",
    "0.5bar": "0p5bar",
}

HYPER = {
    "batch_size": 64,
    "num_epochs": 1, # 1000 for full training
    "eval_freq": 40,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "step_size": 60,
    "gamma": 0.5,
    "optimizer": "Adam",
    "patience": 100,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train RetNet gas-adsorption model.")
    p.add_argument(
        "--pressure",
        type=str,
        choices=list(PRESSURE_SLUG.keys()),
        default="1bar",
        help="Used in checkpoint filename slug (pick to match training pressure).",
    )
    p.add_argument(
        "--target",
        type=str,
        choices=["Xe", "Kr"],
        default="Xe",
        help="Xe or Kr (CSV column Xe_cm3_per_cm3_value vs Kr_cm3_per_cm3_value).",
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed.")
    p.add_argument(
        "--load-dir",
        type=str,
        default="./grids",
        help="Folder with all_<N>grids CSV (targets / PLD,LCD)",
    )
    p.add_argument(
        "--grid-path",
        type=str,
        default="./sample_data",
        help="Split roots: grid-path/train, val, test with clean.{json,npy}",
    )
    p.add_argument(
        "--model-save-dir",
        type=str,
        default="./models",
        help="Directory for *.pt checkpoints",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="pred",
        help="CSV outputs (predictions on test)",
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


def get_extras_list(df_all: pd.DataFrame, batch_dir: str):
    json_path = os.path.join(batch_dir, "clean.json")
    with open(json_path, "r") as f:
        names = json.load(f)["name"]
    return df_all.loc[names, ["PLD", "LCD"]].values.astype(np.float32)


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    args = parse_args()

    pressure_str = PRESSURE_SLUG[args.pressure]
    target_col = TARGET_COLUMNS[args.target]
    set_seed(args.seed)

    index_col = "sample"
    grid_type = ""
    num_grids = 3
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    csv_path = os.path.join(
        args.load_dir, f"all_{num_grids}grids_{grid_type}".strip("_") + ".csv"
    )
    grid_dir = args.grid_path
    train_dir = os.path.join(grid_dir, "train")
    val_dir = os.path.join(grid_dir, "val")
    test_dir = os.path.join(grid_dir, "test")

    model_name = (
        f"MixNewSplit{pressure_str}_{num_grids}_grids_{grid_type}_{target_col}_"
        f"seed_{args.seed}_no_64_{timestamp}_"
        f"{HYPER['batch_size']}_{HYPER['learning_rate']}_{HYPER['weight_decay']}_"
        f"{HYPER['step_size']}_{HYPER['gamma']}_{HYPER['optimizer']}"
    )

    logger = setup_logger("./log", f"{model_name}_{timestamp}.log")
    logger.info("Target column: %s", target_col)
    logger.info("Hyperparameters: %s", HYPER)
    logger.info("csv_path: %s", csv_path)

    df_all = pd.read_csv(csv_path, index_col=index_col)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    X_train, y_train = load_data(train_dir, csv_path, target_col, index_col)
    X_val, y_val = load_data(val_dir, csv_path, target_col, index_col)
    X_test, y_test = load_data(test_dir, csv_path, target_col, index_col)

    logger.info("X_train %s y_train %s", X_train.shape, y_train.shape)
    logger.info("X_val %s y_val %s", X_val.shape, y_val.shape)
    logger.info("X_test %s y_test %s", X_test.shape, y_test.shape)

    extras_train = get_extras_list(df_all, train_dir)
    extras_val = get_extras_list(df_all, val_dir)
    extras_test = get_extras_list(df_all, test_dir)

    assert not np.isnan(extras_train).any()
    assert not np.isnan(extras_val).any()
    assert not np.isnan(extras_test).any()

    if grid_type != "concat":
        X_train = X_train.reshape(
            -1, 1, X_train.shape[1], X_train.shape[2], X_train.shape[3]
        )
        X_val = X_val.reshape(-1, 1, X_val.shape[1], X_val.shape[2], X_val.shape[3])
        X_test = X_test.reshape(-1, 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    else:
        X_train = np.transpose(X_train, (0, 4, 1, 2, 3))
        X_val = np.transpose(X_val, (0, 4, 1, 2, 3))
        X_test = np.transpose(X_test, (0, 4, 1, 2, 3))

    property_cols = ["PLD", "LCD"]
    # Flat or synthetic tensors can have σ=0 and break torchvision Normalize.
    tr_std = max(float(X_train.std()), 1e-6)
    va_std = max(float(X_val.std()), 1e-6)
    te_std = max(float(X_test.std()), 1e-6)
    std_train = transforms.Normalize(mean=X_train.mean(), std=tr_std)
    std_val = transforms.Normalize(mean=X_val.mean(), std=va_std)
    std_test = transforms.Normalize(mean=X_test.mean(), std=te_std)

    train_ds = CustomDataset(X=X_train, y=y_train, extras=extras_train, transform_X=std_train)
    val_ds = CustomDataset(X=X_val, y=y_val, extras=extras_val, transform_X=std_val)
    test_ds = CustomDataset(X=X_test, y=y_test, extras=extras_test, transform_X=std_test)

    bs_train = HYPER["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        pin_memory=True,
        # Avoid empty loader when dataset is smaller than a single batch (e.g. demos).
        drop_last=len(train_ds) >= bs_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2 * HYPER["batch_size"],
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=2 * HYPER["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    ch = 3 if grid_type == "concat" else 1
    net = RetNet(extra_dim=len(property_cols), channel=ch).to(device)

    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(
        net.parameters(),
        lr=HYPER["learning_rate"],
        weight_decay=HYPER["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=HYPER["step_size"],
        gamma=HYPER["gamma"],
        verbose=True,
    )

    net.apply(lambda m: init_weights(m, a=0.01))
    with torch.no_grad():
        net.fc2[-1].bias.fill_(y_train.mean())
    logger.info("Initialized final bias to %.4f", float(y_train.mean()))

    lm = LearningMethod(network=net, optimizer=optimizer, criterion=criterion, logger=logger)
    lm.train(
        train_loader=train_loader,
        val_loader=val_loader,
        val_loss_freq=HYPER["eval_freq"],
        epochs=HYPER["num_epochs"],
        scheduler=scheduler,
        metric=r2_score,
        device=device,
        verbose=True,
        patience=HYPER["patience"],
    )

    lm.net.eval()
    preds, truths = [], []
    for xb, yb, ex in test_loader:
        xb, ex = xb.to(device), ex.to(device)
        with torch.no_grad():
            preds.append(lm.predict(xb, ex).cpu())
        truths.append(yb)

    y_pred = torch.cat(preds).numpy().ravel()
    y_true = torch.cat(truths).numpy().ravel()
    logger.info(
        "Test R^2 %.6f MSE %.6f MAE %.6f",
        r2_score(torch.tensor(y_pred), torch.tensor(y_true)).item(),
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
    )

    os.makedirs(args.model_save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.model_save_dir, f"{model_name}__3features_state_dict.pt"
    )
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": HYPER["num_epochs"],
        },
        ckpt_path,
    )
    logger.info("Saved checkpoint to %s", ckpt_path)

    os.makedirs(args.output_dir, exist_ok=True)
    pred_csv = os.path.join(
        args.output_dir, f"predictions_{model_name}_3features.csv"
    )
    pd.DataFrame({"true_value": y_true, "predicted_value": y_pred}).to_csv(
        pred_csv, index=False
    )
    logger.info("Predictions saved to %s", pred_csv)


if __name__ == "__main__":
    main()
