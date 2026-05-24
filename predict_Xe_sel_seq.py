"""Derive per-pressure Xe selectivity from paired sequence prediction CSVs.

Output CSV contains: id, per-pressure Kr/Xe true and predicted adsorption columns,
sel_true_<pressure>, sel_pred_<pressure> for each pressure level.

Example::

  python predict_Xe_sel_seq.py \
    --kr-pred /path/to/kr_seq_predictions.csv \
    --xe-pred /path/to/xe_seq_predictions.csv \
    --output-csv /path/to/xe_selectivity_seq.csv
"""

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRESSURE_TAGS = ["0p1bar", "0p25bar", "0p5bar", "1bar", "5bar", "10bar"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Derived sequence Xe selectivity from paired Kr/Xe wide CSVs."
    )
    p.add_argument("--kr-pred", type=str, required=True, help="Kr run (wide CSV)")
    p.add_argument("--xe-pred", type=str, required=True, help="Xe run (wide CSV)")
    p.add_argument("--id-col", type=str, default="mof", help="Merge key")
    p.add_argument(
        "--mix-xe",
        type=float,
        default=0.2,
        help="Mole fraction in denominator for Xe",
    )
    p.add_argument(
        "--mix-kr",
        type=float,
        default=0.8,
        help="Mole fraction in denominator for Kr",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="predictions/xe_selectivity_seq.csv",
        help="Output CSV path for merged adsorption and derived per-pressure Xe selectivity columns",
    )
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: list, path: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def derived_selectivity(
    xe: pd.Series,
    kr: pd.Series,
    xe_mix: float,
    kr_mix: float,
) -> pd.Series:
    xe = xe.astype(float)
    kr = kr.astype(float)
    out = (xe / xe_mix) / (kr / kr_mix)
    both_zero = xe.eq(0) & kr.eq(0)
    out = out.astype(float)
    out.loc[both_zero] = 0.0
    return out


def main():
    args = parse_args()

    kr_path = args.kr_pred
    xe_path = args.xe_pred

    suffix_kr = "_kr"
    suffix_xe = "_xe"

    df_k = pd.read_csv(kr_path)
    df_x = pd.read_csv(xe_path)

    need_k = [args.id_col] + [f"true_{t}" for t in PRESSURE_TAGS] + [f"pred_{t}" for t in PRESSURE_TAGS]
    need_x = [args.id_col] + [f"true_{t}" for t in PRESSURE_TAGS] + [f"pred_{t}" for t in PRESSURE_TAGS]

    _require_cols(df_k, need_k, kr_path)
    _require_cols(df_x, need_x, xe_path)

    df_k_sub = df_k[[args.id_col] + [f"true_{t}" for t in PRESSURE_TAGS] + [f"pred_{t}" for t in PRESSURE_TAGS]].copy()
    df_x_sub = df_x[[args.id_col] + [f"true_{t}" for t in PRESSURE_TAGS] + [f"pred_{t}" for t in PRESSURE_TAGS]].copy()

    df_all = pd.merge(
        df_k_sub,
        df_x_sub,
        on=args.id_col,
        how="inner",
        suffixes=(suffix_kr, suffix_xe),
    )

    pooled_true = []
    pooled_pred = []

    print("Derived Xe selectivity (Kr/Xe predictions → selectivity)")
    for p in PRESSURE_TAGS:
        tkr = f"true_{p}{suffix_kr}"
        tkx = f"true_{p}{suffix_xe}"
        pkr = f"pred_{p}{suffix_kr}"
        pkx = f"pred_{p}{suffix_xe}"

        xe_t = df_all[tkx]
        kr_t = df_all[tkr]
        xe_p = df_all[pkx]
        kr_p = df_all[pkr]

        st = derived_selectivity(xe_t, kr_t, args.mix_xe, args.mix_kr)
        sp = derived_selectivity(xe_p, kr_p, args.mix_xe, args.mix_kr)

        df_all[f"sel_true_{p}"] = st
        df_all[f"sel_pred_{p}"] = sp

        yt = st.values.astype(float)
        yp = sp.values.astype(float)
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = math.sqrt(mean_squared_error(yt, yp))
        print(f"  [{p}] R²: {r2:.6f}  MAE: {mae:.6f}  RMSE: {rmse:.6f}")

        pooled_true.append(st)
        pooled_pred.append(sp)

    y_true_overall = pd.concat(pooled_true, axis=0, ignore_index=True).values.astype(float)
    y_pred_overall = pd.concat(pooled_pred, axis=0, ignore_index=True).values.astype(float)

    r2o = r2_score(y_true_overall, y_pred_overall)
    mae_o = mean_absolute_error(y_true_overall, y_pred_overall)
    rmse_o = math.sqrt(mean_squared_error(y_true_overall, y_pred_overall))

    print(
        "  [all_pressures pooled] "
        f"R²: {r2o:.6f}  MAE: {mae_o:.6f}  RMSE: {rmse_o:.6f}"
    )
    print(f"Rows merged on {args.id_col!r}: {len(df_all)}")

    out_path = args.output_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
