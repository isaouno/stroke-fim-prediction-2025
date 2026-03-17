# fim_fim_predict.py
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


FIM_M_BOUNDS = (13, 91)
FIM_C_BOUNDS = (5, 35)
FIM_T_BOUNDS = (18, 126)

REQUIRED_COLS = [
    "age",
    "onset_to_admit_days",
    "fim_admit_m",
    "fim_admit_c",
    "fim_discharge_m",
    "fim_discharge_c",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    y_col: str
    x_cols: List[str]
    kind: str  # "m", "c", "t"


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def round_half_up(x: np.ndarray) -> np.ndarray:
    # half-up rounding for non-negative values
    return np.floor(x + 0.5)


def apply_constraints_mc(pred_m: np.ndarray, pred_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pm = round_half_up(pred_m).astype(float)
    pc = round_half_up(pred_c).astype(float)
    pm = np.clip(pm, FIM_M_BOUNDS[0], FIM_M_BOUNDS[1])
    pc = np.clip(pc, FIM_C_BOUNDS[0], FIM_C_BOUNDS[1])
    pt = pm + pc  # sum-consistency priority
    return pm, pc, pt


def apply_constraints_total(pred_t: np.ndarray) -> np.ndarray:
    pt = round_half_up(pred_t).astype(float)
    pt = np.clip(pt, FIM_T_BOUNDS[0], FIM_T_BOUNDS[1])
    return pt


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        val = float(r2_score(y_true, y_pred))
        if np.isfinite(val):
            return val
        return float("nan")
    except Exception:
        return float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = safe_r2(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def calibration_intercept_slope(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # y = a + b * pred
    if np.all(~np.isfinite(y_true)) or np.all(~np.isfinite(y_pred)):
        return float("nan"), float("nan")
    if np.nanstd(y_pred) == 0:
        return float("nan"), float("nan")

    X = np.column_stack([np.ones_like(y_pred), y_pred])
    try:
        beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        return a, b
    except Exception:
        return float("nan"), float("nan")


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LinearRegression()),
        ]
    )


def oof_predict(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    constraints_kind: Optional[str] = None,
    constraints_total_direct: bool = False,
) -> np.ndarray:
    """
    constraints_kind:
      - None: no constraints
      - "m" or "c": apply rounding+clip to the predicted y (for that model)
      - "t": apply rounding+clip to total direct prediction only if constraints_total_direct=True
    """
    n = len(y)
    preds = np.full(shape=(n,), fill_value=np.nan, dtype=float)

    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for tr_idx, te_idx in cv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]

        model = make_pipeline()
        model.fit(X_tr, y_tr)
        p = model.predict(X_te).astype(float)

        if constraints_kind in ("m", "c"):
            if constraints_kind == "m":
                p = round_half_up(p)
                p = np.clip(p, FIM_M_BOUNDS[0], FIM_M_BOUNDS[1])
            else:
                p = round_half_up(p)
                p = np.clip(p, FIM_C_BOUNDS[0], FIM_C_BOUNDS[1])
        elif constraints_kind == "t" and constraints_total_direct:
            p = apply_constraints_total(p)

        preds[te_idx] = p

    return preds


def bootstrap_ci_on_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    B: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    stats = {
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "calib_a": [],
        "calib_b": [],
    }

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        m = compute_metrics(yt, yp)
        a, b = calibration_intercept_slope(yt, yp)

        stats["MAE"].append(m["MAE"])
        stats["RMSE"].append(m["RMSE"])
        stats["R2"].append(m["R2"])
        stats["calib_a"].append(a)
        stats["calib_b"].append(b)

    rows = []
    for k, arr in stats.items():
        arr = np.asarray(arr, dtype=float)
        point = float(np.nanmedian(arr))
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        rows.append({"metric": k, "point": point, "ci_low": float(lo), "ci_high": float(hi), "B": int(B)})

    return pd.DataFrame(rows)


def repeated_cv_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    seeds: List[int],
    constraints_kind: Optional[str] = None,
    constraints_total_direct: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      - perf_df: per-seed metrics (MAE/RMSE/R2)
      - pred_mat: shape (n_subjects, n_repeats) with OOF predictions for each repeat
    """
    n = len(y)
    k = len(seeds)
    pred_mat = np.full((n, k), np.nan, dtype=float)

    perf_rows = []
    for j, s in enumerate(seeds):
        preds = oof_predict(
            X=X,
            y=y,
            n_splits=5,
            shuffle=True,
            random_state=s,
            constraints_kind=constraints_kind,
            constraints_total_direct=constraints_total_direct,
        )
        pred_mat[:, j] = preds
        met = compute_metrics(y.to_numpy(dtype=float), preds)
        perf_rows.append({"seed": s, **met})

    perf_df = pd.DataFrame(perf_rows)
    perf_df["MAE_mean"] = perf_df["MAE"].mean()
    perf_df["MAE_sd"] = perf_df["MAE"].std(ddof=1)

    return perf_df, pred_mat


def icc_2_1(pred_mat: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, absolute agreement, single measurement.
    pred_mat: n_subjects x k_raters (here: repeats)
    """
    Y = np.asarray(pred_mat, dtype=float)
    if Y.ndim != 2:
        return float("nan")

    mask = np.all(np.isfinite(Y), axis=1)
    Y = Y[mask]
    n, k = Y.shape
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = np.mean(Y)
    row_means = np.mean(Y, axis=1, keepdims=True)
    col_means = np.mean(Y, axis=0, keepdims=True)

    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)
    SSE = np.sum((Y - row_means - col_means + grand_mean) ** 2)

    dfR = n - 1
    dfC = k - 1
    dfE = (n - 1) * (k - 1)

    MSR = SSR / dfR if dfR > 0 else np.nan
    MSC = SSC / dfC if dfC > 0 else np.nan
    MSE = SSE / dfE if dfE > 0 else np.nan

    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")

    icc = (MSR - MSE) / denom
    return float(icc)


def bootstrap_icc_ci(pred_mat: np.ndarray, B: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    Y = np.asarray(pred_mat, dtype=float)

    mask = np.all(np.isfinite(Y), axis=1)
    Y = Y[mask]
    n = Y.shape[0]
    if n < 2:
        return float("nan"), float("nan"), float("nan")

    point = icc_2_1(Y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        vals.append(icc_2_1(Y[idx, :]))
    vals = np.asarray(vals, dtype=float)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def optimism_correction(
    X: pd.DataFrame,
    y: pd.Series,
    B: int,
    seed: int,
    constraints_kind: Optional[str] = None,
    constraints_total_direct: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(y)

    full_model = make_pipeline()
    full_model.fit(X, y)
    pred_full = full_model.predict(X).astype(float)

    if constraints_kind in ("m", "c"):
        if constraints_kind == "m":
            pred_full = np.clip(round_half_up(pred_full), FIM_M_BOUNDS[0], FIM_M_BOUNDS[1])
        else:
            pred_full = np.clip(round_half_up(pred_full), FIM_C_BOUNDS[0], FIM_C_BOUNDS[1])
    elif constraints_kind == "t" and constraints_total_direct:
        pred_full = apply_constraints_total(pred_full)

    apparent = compute_metrics(y.to_numpy(dtype=float), pred_full)

    opt_mae, opt_rmse, opt_r2 = [], [], []
    X_np = X.reset_index(drop=True)
    y_np = y.reset_index(drop=True)

    base_model = make_pipeline()
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        X_b = X_np.iloc[idx]
        y_b = y_np.iloc[idx]

        m = clone(base_model)
        m.fit(X_b, y_b)

        pred_b_in = m.predict(X_b).astype(float)
        pred_b_out = m.predict(X_np).astype(float)

        if constraints_kind in ("m", "c"):
            if constraints_kind == "m":
                pred_b_in = np.clip(round_half_up(pred_b_in), FIM_M_BOUNDS[0], FIM_M_BOUNDS[1])
                pred_b_out = np.clip(round_half_up(pred_b_out), FIM_M_BOUNDS[0], FIM_M_BOUNDS[1])
            else:
                pred_b_in = np.clip(round_half_up(pred_b_in), FIM_C_BOUNDS[0], FIM_C_BOUNDS[1])
                pred_b_out = np.clip(round_half_up(pred_b_out), FIM_C_BOUNDS[0], FIM_C_BOUNDS[1])
        elif constraints_kind == "t" and constraints_total_direct:
            pred_b_in = apply_constraints_total(pred_b_in)
            pred_b_out = apply_constraints_total(pred_b_out)

        met_in = compute_metrics(y_b.to_numpy(dtype=float), pred_b_in)
        met_out = compute_metrics(y_np.to_numpy(dtype=float), pred_b_out)

        opt_mae.append(met_in["MAE"] - met_out["MAE"])
        opt_rmse.append(met_in["RMSE"] - met_out["RMSE"])
        opt_r2.append(met_in["R2"] - met_out["R2"])

    opt = {
        "MAE": float(np.nanmean(opt_mae)),
        "RMSE": float(np.nanmean(opt_rmse)),
        "R2": float(np.nanmean(opt_r2)),
    }

    corrected = {
        "MAE": float(apparent["MAE"] - opt["MAE"]),
        "RMSE": float(apparent["RMSE"] - opt["RMSE"]),
        "R2": float(apparent["R2"] - opt["R2"]) if np.isfinite(apparent["R2"]) and np.isfinite(opt["R2"]) else float("nan"),
    }

    rows = []
    for metric in ["MAE", "RMSE", "R2"]:
        rows.append(
            {
                "metric": metric,
                "apparent_full": apparent[metric],
                "optimism_mean": opt[metric],
                "corrected": corrected[metric],
                "B_opt": int(B),
            }
        )
    return pd.DataFrame(rows)


def load_excel(path: str, sheet: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"入力Excelが見つかりません: {path}")

    try:
        if sheet is None or str(sheet).strip() == "":
            df = pd.read_excel(path, sheet_name=0)
        else:
            df = pd.read_excel(path, sheet_name=sheet)
    except ValueError as e:
        raise ValueError(f"sheet名が不正です: {sheet} / {e}") from e
    except Exception as e:
        raise RuntimeError(f"Excel読み込みに失敗しました: {e}") from e

    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def qc_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    rows.append({"item": "n_rows", "value": n})
    rows.append({"item": "missing_required_columns", "value": ", ".join(missing) if missing else ""})

    if missing:
        return pd.DataFrame(rows)

    df_num = coerce_numeric(df, REQUIRED_COLS + ["fim_admit_total", "fim_discharge_total"])
    for c in REQUIRED_COLS + ["fim_admit_total", "fim_discharge_total"]:
        rows.append({"item": f"missing_or_non_numeric_{c}", "value": int(df_num[c].isna().sum())})

    def count_out_of_range(x: pd.Series, lo: int, hi: int) -> int:
        x = pd.to_numeric(x, errors="coerce")
        return int(((x < lo) | (x > hi)).sum(skipna=True))

    rows.append({"item": "out_of_range_fim_admit_m", "value": count_out_of_range(df_num["fim_admit_m"], *FIM_M_BOUNDS)})
    rows.append({"item": "out_of_range_fim_admit_c", "value": count_out_of_range(df_num["fim_admit_c"], *FIM_C_BOUNDS)})
    rows.append({"item": "out_of_range_fim_admit_total", "value": count_out_of_range(df_num["fim_admit_total"], *FIM_T_BOUNDS)})

    rows.append({"item": "out_of_range_fim_discharge_m", "value": count_out_of_range(df_num["fim_discharge_m"], *FIM_M_BOUNDS)})
    rows.append({"item": "out_of_range_fim_discharge_c", "value": count_out_of_range(df_num["fim_discharge_c"], *FIM_C_BOUNDS)})
    rows.append({"item": "out_of_range_fim_discharge_total", "value": count_out_of_range(df_num["fim_discharge_total"], *FIM_T_BOUNDS)})

    admit_sum = df_num["fim_admit_m"] + df_num["fim_admit_c"]
    discharge_sum = df_num["fim_discharge_m"] + df_num["fim_discharge_c"]
    rows.append({"item": "admit_total_mismatch_count", "value": int((admit_sum != df_num["fim_admit_total"]).sum(skipna=True))})
    rows.append({"item": "discharge_total_mismatch_count", "value": int((discharge_sum != df_num["fim_discharge_total"]).sum(skipna=True))})

    rows.append({"item": "age_negative_count", "value": int((df_num["age"] < 0).sum(skipna=True))})
    rows.append({"item": "onset_to_admit_days_negative_count", "value": int((df_num["onset_to_admit_days"] < 0).sum(skipna=True))})

    return pd.DataFrame(rows)


def build_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="A_mFIM",
            y_col="fim_discharge_m",
            x_cols=["age", "onset_to_admit_days", "fim_admit_m", "fim_admit_c"],
            kind="m",
        ),
        ModelSpec(
            name="B_cFIM",
            y_col="fim_discharge_c",
            x_cols=["age", "onset_to_admit_days", "fim_admit_m", "fim_admit_c"],
            kind="c",
        ),
        ModelSpec(
            name="C_totalFIM",
            y_col="fim_discharge_total",
            x_cols=["age", "onset_to_admit_days", "fim_admit_total"],
            kind="t",
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear regression models for discharge FIM with internal validation.")
    p.add_argument("--input_xlsx", required=True, help="入力Excel(.xlsx)のパス")
    p.add_argument("--sheet", default="", help="シート名（先頭シートなら空でOK）")
    p.add_argument("--outdir", required=True, help="出力フォルダ（results.xlsxを作成）")

    # NOTE: argparse help uses old-style '%' formatting internally; escape '%' as '%%'
    p.add_argument("--B_ci", type=int, default=2000, help="OOF指標95%%CIのブートストラップ反復回数（既定: 2000）")
    p.add_argument("--B_icc", type=int, default=2000, help="ICC(2,1) 95%%CIのブートストラップ反復回数（既定: 2000）")
    p.add_argument("--B_opt", type=int, default=500, help="楽観補正(optimism correction)の反復回数（既定: 500）")

    p.add_argument("--constraints", action="store_true", help="運用制約を適用（m/cを整数化+clipし、totalはm+c）")
    p.add_argument("--constraints_total", action="store_true", help="モデルC（total直予測）にも整数化+clipを適用")

    p.add_argument("--seed_ci", type=int, default=4242, help="CIブートストラップの乱数seed（既定: 4242）")
    p.add_argument("--seed_icc", type=int, default=4343, help="ICCブートストラップの乱数seed（既定: 4343）")
    p.add_argument("--seed_opt", type=int, default=4444, help="楽観補正ブートストラップの乱数seed（既定: 4444）")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    out_xlsx = os.path.join(outdir, "results.xlsx")

    try:
        df_raw = load_excel(args.input_xlsx, args.sheet)
    except Exception as e:
        eprint(str(e))
        return 1

    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        eprint("必要な列が見つかりません: " + ", ".join(missing))
        eprint("現在の列名: " + ", ".join(map(str, list(df_raw.columns))))
        return 1

    df = df_raw.copy()
    df["fim_admit_total"] = pd.to_numeric(df["fim_admit_m"], errors="coerce") + pd.to_numeric(df["fim_admit_c"], errors="coerce")
    df["fim_discharge_total"] = pd.to_numeric(df["fim_discharge_m"], errors="coerce") + pd.to_numeric(df["fim_discharge_c"], errors="coerce")

    qc_df = qc_table(df)

    oof_pred_df = pd.DataFrame({"row_id": np.arange(len(df)) + 1})
    oof_perf_rows = []
    ci_rows = []
    repcv_rows_all = []
    per_subject_sd_rows = []
    icc_rows = []
    opt_rows_all = []

    oof_pred_df["y_true_m"] = pd.to_numeric(df["fim_discharge_m"], errors="coerce")
    oof_pred_df["y_true_c"] = pd.to_numeric(df["fim_discharge_c"], errors="coerce")
    oof_pred_df["y_true_total"] = pd.to_numeric(df["fim_discharge_total"], errors="coerce")

    specs = build_specs()

    # Primary CV (seed=42)
    for spec in specs:
        X = df[spec.x_cols].copy()
        y = pd.to_numeric(df[spec.y_col], errors="coerce")

        ck = None
        c_total_direct = False
        if args.constraints and spec.kind in ("m", "c"):
            ck = spec.kind
        if args.constraints_total and spec.kind == "t":
            ck = "t"
            c_total_direct = True

        preds = oof_predict(
            X=X,
            y=y,
            n_splits=5,
            shuffle=True,
            random_state=42,
            constraints_kind=ck,
            constraints_total_direct=c_total_direct,
        )

        col_pred = f"oof_pred_{spec.name}"
        oof_pred_df[col_pred] = preds

        mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(preds)
        yt = y.to_numpy(dtype=float)[mask]
        yp = preds[mask]

        met = compute_metrics(yt, yp)
        a, b = calibration_intercept_slope(yt, yp)

        oof_perf_rows.append(
            {
                "model": spec.name,
                "outcome": spec.y_col,
                "constraints_applied": bool((args.constraints and spec.kind in ("m", "c")) or (args.constraints_total and spec.kind == "t")),
                **met,
                "calib_a": a,
                "calib_b": b,
                "n_eval": int(mask.sum()),
            }
        )

        ci_df = bootstrap_ci_on_oof(yt, yp, B=int(args.B_ci), seed=int(args.seed_ci) + hash(spec.name) % 100000)
        ci_df.insert(0, "model", spec.name)
        ci_df.insert(1, "outcome", spec.y_col)
        ci_rows.append(ci_df)

    if "oof_pred_A_mFIM" in oof_pred_df.columns and "oof_pred_B_cFIM" in oof_pred_df.columns:
        pm = oof_pred_df["oof_pred_A_mFIM"].to_numpy(dtype=float)
        pc = oof_pred_df["oof_pred_B_cFIM"].to_numpy(dtype=float)
        oof_pred_df["oof_pred_total_from_AplusB"] = pm + pc

    rep_seeds = list(range(1000, 1040))
    for spec in specs:
        X = df[spec.x_cols].copy()
        y = pd.to_numeric(df[spec.y_col], errors="coerce")

        ck = None
        c_total_direct = False
        if args.constraints and spec.kind in ("m", "c"):
            ck = spec.kind
        if args.constraints_total and spec.kind == "t":
            ck = "t"
            c_total_direct = True

        perf_df, pred_mat = repeated_cv_predictions(
            X=X,
            y=y,
            seeds=rep_seeds,
            constraints_kind=ck,
            constraints_total_direct=c_total_direct,
        )
        perf_df.insert(0, "model", spec.name)
        perf_df.insert(1, "outcome", spec.y_col)
        perf_df.insert(2, "constraints_applied", bool((args.constraints and spec.kind in ("m", "c")) or (args.constraints_total and spec.kind == "t")))
        repcv_rows_all.append(perf_df)

        sd = np.nanstd(pred_mat, axis=1, ddof=1)
        mu = np.nanmean(pred_mat, axis=1)
        per_subject_sd_rows.append(
            pd.DataFrame(
                {
                    "row_id": np.arange(len(df)) + 1,
                    "model": spec.name,
                    "outcome": spec.y_col,
                    "pred_mean_40": mu,
                    "pred_sd_40": sd,
                }
            )
        )

        icc_point, icc_lo, icc_hi = bootstrap_icc_ci(pred_mat, B=int(args.B_icc), seed=int(args.seed_icc) + hash(spec.name) % 100000)
        icc_rows.append(
            {
                "model": spec.name,
                "outcome": spec.y_col,
                "constraints_applied": bool((args.constraints and spec.kind in ("m", "c")) or (args.constraints_total and spec.kind == "t")),
                "ICC_2_1": icc_point,
                "CI_low": icc_lo,
                "CI_high": icc_hi,
                "B_icc": int(args.B_icc),
                "n_subjects_used": int(np.all(np.isfinite(pred_mat), axis=1).sum()),
                "k_repeats": int(pred_mat.shape[1]),
            }
        )

    for spec in specs:
        X = df[spec.x_cols].copy()
        y = pd.to_numeric(df[spec.y_col], errors="coerce")

        ck = None
        c_total_direct = False
        if args.constraints and spec.kind in ("m", "c"):
            ck = spec.kind
        if args.constraints_total and spec.kind == "t":
            ck = "t"
            c_total_direct = True

        opt_df = optimism_correction(
            X=X,
            y=y,
            B=int(args.B_opt),
            seed=int(args.seed_opt) + hash(spec.name) % 100000,
            constraints_kind=ck,
            constraints_total_direct=c_total_direct,
        )
        opt_df.insert(0, "model", spec.name)
        opt_df.insert(1, "outcome", spec.y_col)
        opt_df.insert(2, "constraints_applied", bool((args.constraints and spec.kind in ("m", "c")) or (args.constraints_total and spec.kind == "t")))
        opt_rows_all.append(opt_df)

    oof_perf_df = pd.DataFrame(oof_perf_rows)
    ci_df_all = pd.concat(ci_rows, axis=0, ignore_index=True) if ci_rows else pd.DataFrame()
    repcv_df_all = pd.concat(repcv_rows_all, axis=0, ignore_index=True) if repcv_rows_all else pd.DataFrame()
    per_subject_sd_df = pd.concat(per_subject_sd_rows, axis=0, ignore_index=True) if per_subject_sd_rows else pd.DataFrame()
    icc_df = pd.DataFrame(icc_rows)
    opt_df_all = pd.concat(opt_rows_all, axis=0, ignore_index=True) if opt_rows_all else pd.DataFrame()

    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            qc_df.to_excel(w, sheet_name="QC", index=False)
            oof_pred_df.to_excel(w, sheet_name="OOF_predictions", index=False)
            oof_perf_df.to_excel(w, sheet_name="OOF_performance", index=False)
            ci_df_all.to_excel(w, sheet_name="CI", index=False)
            repcv_df_all.to_excel(w, sheet_name="RepeatedCV", index=False)
            per_subject_sd_df.to_excel(w, sheet_name="PerSubject_SD", index=False)
            icc_df.to_excel(w, sheet_name="ICC", index=False)
            opt_df_all.to_excel(w, sheet_name="OptimismCorrection", index=False)
    except Exception as e:
        eprint(f"results.xlsx の書き込みに失敗しました: {e}")
        return 1

    print(f"完了: {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
