"""Derive Xe selectivity from two predict_gas_ads CSVs (Kr and Xe adsorption predictions).

Xe_selectivity = (Xe_true/mix_xe) / (Kr_true/mix_kr); predicted version uses *_pred columns.
Default mixing ratios: 20% Xe / 80% Kr.

Output CSV contains: id, Kr/Xe true and predicted adsorption columns, Xe_selectivity, Xe_selectivity_pred.

Example::

  python predict_Xe_sel.py \
    --kr-pred /path/to/kr_predictions.csv \
    --xe-pred /path/to/xe_predictions.csv \
    --output-csv /path/to/xe_selectivity.csv
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description="Derived Xe selectivity from Kr and Xe prediction CSVs.")
    p.add_argument(
        "--kr-pred",
        type=str,
        required=True,
        help="CSV from predict_sel (Kr target)",
    )
    p.add_argument(
        "--xe-pred",
        type=str,
        required=True,
        help="CSV from predict_sel (Xe target)",
    )
    p.add_argument("--id-col", type=str, default="mof", help="Merge key (default: mof)")
    p.add_argument("--xe-col", type=str, default="Xe_cm3_per_cm3_value")
    p.add_argument("--kr-col", type=str, default="Kr_cm3_per_cm3_value")
    p.add_argument(
        "--mix-xe",
        type=float,
        default=0.2,
        help="Denominator mole fraction used for Xe in selectivity formula",
    )
    p.add_argument(
        "--mix-kr",
        type=float,
        default=0.8,
        help="Denominator mole fraction used for Kr in selectivity formula",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="predictions/xe_selectivity.csv",
        help="Output CSV path for merged adsorption and derived Xe selectivity columns",
    )
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: list, path: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; have {list(df.columns)}")


def derived_selectivity(
    xe: pd.Series,
    kr: pd.Series,
    xe_mix: float,
    kr_mix: float,
) -> pd.Series:
    out = (xe.astype(float) / xe_mix) / (kr.astype(float) / kr_mix)
    both_zero = (xe.astype(float).eq(0)) & (kr.astype(float).eq(0))
    out = out.astype(float)
    out.loc[both_zero] = 0.0
    return out


def main():
    args = parse_args()

    xe_pred_col = f"{args.xe_col}_pred"
    kr_pred_col = f"{args.kr_col}_pred"

    df_kr = pd.read_csv(args.kr_pred)
    df_xe = pd.read_csv(args.xe_pred)

    _require_cols(
        df_kr,
        [args.id_col, args.kr_col, kr_pred_col],
        args.kr_pred,
    )
    _require_cols(
        df_xe,
        [args.id_col, args.xe_col, xe_pred_col],
        args.xe_pred,
    )

    df_kr_s = df_kr[[args.id_col, args.kr_col, kr_pred_col]].copy()
    df_xe_s = df_xe[[args.id_col, args.xe_col, xe_pred_col]].copy()
    df_all = df_kr_s.merge(df_xe_s, on=args.id_col, how="inner")

    xe_true = df_all[args.xe_col]
    kr_true = df_all[args.kr_col]

    xe_p = df_all[xe_pred_col]
    kr_p = df_all[kr_pred_col]

    df_all["Xe_selectivity"] = derived_selectivity(xe_true, kr_true, args.mix_xe, args.mix_kr)
    df_all["Xe_selectivity_pred"] = derived_selectivity(xe_p, kr_p, args.mix_xe, args.mix_kr)

    y_true = df_all["Xe_selectivity"].astype(float).values
    y_pred = df_all["Xe_selectivity_pred"].astype(float).values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    print(f"Derived Xe selectivity — R²: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    print(f"Rows after inner merge on {args.id_col!r}: {len(df_all)}")

    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
