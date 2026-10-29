#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V17: Structured Spectral Ablation + Covariance-Aware PCA

Input: patient_bandpower_features.csv (129 patients, 190 patient-level features)
Models: Elastic-Net logistic regression and linear SVM
Protocol: repeated patient-level nested CV (outer 5-fold, inner 4-fold)
Primary metric: mean outer-fold ROC-AUC

Locked feature modes:
  base190        all 190 features
  abs95          absolute power only
  rel95          relative power only
  alpha38        alpha absolute + relative
  theta_alpha76  theta + alpha
  alpha_beta76   alpha + beta
  pca90          all 190 -> StandardScaler -> PCA 90% variance
  pca95          all 190 -> StandardScaler -> PCA 95% variance
  pca_tuned      PCA {90%,95%,99%} selected inside inner CV

All imputation, scaling, PCA and model tuning are fitted on training folds only.
Outer-test is never used for preprocessing or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_ENET = "logreg_elasticnet"
MODEL_SVM = "svm_linear"
MODEL_INDEX = {MODEL_ENET: 2, MODEL_SVM: 3}
LOCKED_MODES = [
    "base190", "abs95", "rel95", "alpha38", "theta_alpha76",
    "alpha_beta76", "pca90", "pca95", "pca_tuned",
]
BANDS = ["delta", "theta", "alpha", "beta", "low_gamma"]


def log(msg: str) -> None:
    print(f"[V17-PCA] {msg}", flush=True)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def validate_table(df: pd.DataFrame) -> List[str]:
    if not {"patient_id", "label"}.issubset(df.columns):
        raise ValueError("feature CSV must contain patient_id and label")
    if df.patient_id.astype(str).duplicated().any():
        raise ValueError("duplicated patient_id")
    if sorted(df.label.astype(int).unique().tolist()) != [0, 1]:
        raise ValueError("labels must be 0/1")

    cols = [c for c in df.columns if c not in {"patient_id", "label"}]
    abs_cols = [c for c in cols if c.startswith("absdb_")]
    rel_cols = [c for c in cols if c.startswith("rel_")]
    if len(cols) != 190 or len(abs_cols) != 95 or len(rel_cols) != 95:
        raise ValueError(
            f"expected 190 features = 95 absdb + 95 rel; got "
            f"{len(cols)} = {len(abs_cols)} + {len(rel_cols)}"
        )
    return cols


def mode_columns(feature_cols: List[str], mode: str) -> List[str]:
    if mode in {"base190", "pca90", "pca95", "pca_tuned"}:
        return feature_cols.copy()
    if mode == "abs95":
        return [c for c in feature_cols if c.startswith("absdb_")]
    if mode == "rel95":
        return [c for c in feature_cols if c.startswith("rel_")]

    def bands(names: List[str]) -> List[str]:
        out = []
        for c in feature_cols:
            if any(
                c.startswith(f"absdb_{b}_") or c.startswith(f"rel_{b}_")
                for b in names
            ):
                out.append(c)
        return out

    if mode == "alpha38":
        return bands(["alpha"])
    if mode == "theta_alpha76":
        return bands(["theta", "alpha"])
    if mode == "alpha_beta76":
        return bands(["alpha", "beta"])
    raise ValueError(f"unknown mode: {mode}")


def make_pipeline(model: str, mode: str, seed: int):
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]

    if mode == "pca90":
        steps.append(("pca", PCA(n_components=0.90, svd_solver="full")))
    elif mode == "pca95":
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    elif mode == "pca_tuned":
        steps.append(("pca", PCA(svd_solver="full")))

    if model == MODEL_ENET:
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight="balanced",
            max_iter=10000,
            random_state=int(seed),
        )
        grid: Dict[str, List[object]] = {
            "clf__C": [0.01, 0.1, 1.0],
            "clf__l1_ratio": [0.0, 0.5, 1.0],
        }
    elif model == MODEL_SVM:
        clf = SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=int(seed),
        )
        grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        raise ValueError(model)

    if mode == "pca_tuned":
        grid["pca__n_components"] = [0.90, 0.95, 0.99]

    steps.append(("clf", clf))
    return Pipeline(steps), grid


def inner_cv(seed: int, fold: int, model: str, n_splits: int):
    # Matches the original baseline's model-specific inner split seed.
    rs = int(seed * 1000 + fold * 17 + MODEL_INDEX[model])
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)


def estimator_seed(seed: int, fold: int, model: str) -> int:
    return int(seed * 10000 + fold * 101 + MODEL_INDEX[model])


def fit_model(X, y, model, mode, seed, fold, inner_splits, n_jobs):
    pipe, grid = make_pipeline(model, mode, estimator_seed(seed, fold, model))
    gs = GridSearchCV(
        pipe,
        grid,
        scoring="roc_auc",
        cv=inner_cv(seed, fold, model, inner_splits),
        refit=True,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    return gs


def positive_probability(model, X):
    p = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    j = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(p[:, j], dtype=float)


def metrics(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def pca_info(fitted):
    if "pca" not in fitted.named_steps:
        return np.nan, np.nan
    p = fitted.named_steps["pca"]
    return int(p.n_components_), float(np.sum(p.explained_variance_ratio_))


def effective_weights(fitted, n_input: int):
    clf = fitted.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return np.full(n_input, np.nan)
    coef = np.asarray(clf.coef_, dtype=float).reshape(-1)
    if "pca" in fitted.named_steps:
        coef = fitted.named_steps["pca"].components_.T @ coef
    if len(coef) != n_input:
        raise RuntimeError("weight dimension mismatch")
    return coef


def summarize_seed(fold_df, pred_df):
    rows = []
    for (seed, mode, model), f in fold_df.groupby(["seed", "mode", "model"]):
        p = pred_df[
            (pred_df.seed == seed)
            & (pred_df["mode"] == mode)
            & (pred_df.model == model)
        ]
        y = p.label.to_numpy(int)
        prob = p.probability.to_numpy(float)
        pred = p.y_pred.to_numpy(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "seed": int(seed),
            "mode": mode,
            "model": model,
            "mean_outer_auc": float(f.roc_auc.mean()),
            "sd_outer_auc": float(f.roc_auc.std(ddof=1)),
            "pooled_oof_auc": float(roc_auc_score(y, prob)),
            "balanced_accuracy_fold_mean": float(f.balanced_accuracy.mean()),
            "macro_f1_fold_mean": float(f.macro_f1.mean()),
            "accuracy_fold_mean": float(f.accuracy.mean()),
            "sensitivity_pooled": float(tp / (tp + fn)),
            "specificity_pooled": float(tn / (tn + fp)),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        })
    return pd.DataFrame(rows)


def summarize_repeated(seed_df):
    rows = []
    metrics_cols = [
        "mean_outer_auc", "pooled_oof_auc", "balanced_accuracy_fold_mean",
        "macro_f1_fold_mean", "accuracy_fold_mean", "sensitivity_pooled",
        "specificity_pooled",
    ]
    for (mode, model), g in seed_df.groupby(["mode", "model"]):
        row = {"mode": mode, "model": model, "n_seeds": int(len(g))}
        for c in metrics_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_outer_auc_mean", "balanced_accuracy_fold_mean_mean"],
        ascending=False,
    ).reset_index(drop=True)


def compare_reference(seed_df, path: Path):
    if path is None or not path.exists():
        return pd.DataFrame()
    ref = pd.read_csv(path)
    need = {
        "seed", "model", "mean_outer_auc", "balanced_accuracy_mean",
        "accuracy_mean", "sensitivity_pooled", "specificity_pooled",
    }
    if not need.issubset(ref.columns):
        return pd.DataFrame()
    m = seed_df.merge(
        ref[[
            "seed", "model", "mean_outer_auc", "balanced_accuracy_mean",
            "accuracy_mean", "sensitivity_pooled", "specificity_pooled",
        ]],
        on=["seed", "model"], how="left", suffixes=("_v17", "_reference")
    )
    m["delta_auc_v17_minus_reference"] = (
        m.mean_outer_auc_v17 - m.mean_outer_auc_reference
    )
    m["delta_ba_v17_minus_reference"] = (
        m.balanced_accuracy_fold_mean - m.balanced_accuracy_mean
    )
    m["delta_accuracy_v17_minus_reference"] = (
        m.accuracy_fold_mean - m.accuracy_mean
    )
    return m


def run(args):
    if args.self_test:
        self_test()
        return

    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_csv.expanduser().resolve())
    feature_cols = validate_table(df)
    df["patient_id"] = df.patient_id.astype(str)
    y = df.label.to_numpy(int)

    modes = LOCKED_MODES if args.modes.lower() == "all" else parse_list(args.modes)
    models = parse_list(args.models)
    seeds = [int(x) for x in parse_list(args.seeds)]

    bad_modes = sorted(set(modes) - set(LOCKED_MODES))
    bad_models = sorted(set(models) - {MODEL_ENET, MODEL_SVM})
    if bad_modes or bad_models:
        raise ValueError(f"bad modes={bad_modes}, bad models={bad_models}")

    expected_dim = {
        "base190": 190, "abs95": 95, "rel95": 95, "alpha38": 38,
        "theta_alpha76": 76, "alpha_beta76": 76,
        "pca90": 190, "pca95": 190, "pca_tuned": 190,
    }
    mode_dims = {}
    for mode in modes:
        n = len(mode_columns(feature_cols, mode))
        if n != expected_dim[mode]:
            raise RuntimeError(f"{mode}: expected {expected_dim[mode]}, got {n}")
        mode_dims[mode] = n

    config = {
        "feature_csv": str(args.feature_csv.expanduser().resolve()),
        "n_patients": int(len(df)),
        "tinnitus_n": int(np.sum(y == 1)),
        "control_n": int(np.sum(y == 0)),
        "modes": modes,
        "mode_input_dimensions": mode_dims,
        "models": models,
        "seeds": seeds,
        "outer_splits": int(args.outer_splits),
        "inner_splits": int(args.inner_splits),
        "primary_metric": "mean outer-fold ROC-AUC",
        "outer_test_used_for_preprocessing_or_tuning": False,
        "pca_tuned_candidates": [0.90, 0.95, 0.99],
    }
    (out / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fold_rows, pred_rows, weight_rows = [], [], []

    for seed in seeds:
        outer = StratifiedKFold(
            n_splits=args.outer_splits, shuffle=True, random_state=seed
        )
        for fold, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y), 1):
            ytr, yte = y[tr], y[te]
            log("=" * 88)
            log(f"seed={seed} fold={fold}/{args.outer_splits} train={len(tr)} test={len(te)}")

            for mode in modes:
                cols = mode_columns(feature_cols, mode)
                X = df[cols].to_numpy(float)
                Xtr, Xte = X[tr], X[te]
                log(f"mode={mode} input_dim={X.shape[1]}")

                for model in models:
                    log(f"model={model}")
                    gs = fit_model(
                        Xtr, ytr, model, mode, seed, fold,
                        args.inner_splits, args.n_jobs
                    )
                    fitted = gs.best_estimator_
                    prob = positive_probability(fitted, Xte)
                    pred = fitted.predict(Xte).astype(int)
                    mm = metrics(yte, prob, pred)
                    ncomp, ev = pca_info(fitted)

                    fold_rows.append({
                        "seed": seed, "fold": fold, "mode": mode, "model": model,
                        "n_train": len(tr), "n_test": len(te),
                        "n_input_features": X.shape[1],
                        "final_inner_best_auc": float(gs.best_score_),
                        "best_params": json.dumps(gs.best_params_, sort_keys=True),
                        "pca_components": ncomp,
                        "pca_explained_variance": ev,
                        **mm,
                    })

                    w = effective_weights(fitted, len(cols))
                    for name, val in zip(cols, w):
                        weight_rows.append({
                            "seed": seed, "fold": fold, "mode": mode,
                            "model": model, "feature": name,
                            "effective_weight_scaled_space": float(val),
                        })

                    for j, idx in enumerate(te):
                        pred_rows.append({
                            "seed": seed, "fold": fold, "mode": mode,
                            "model": model,
                            "patient_id": str(df.iloc[idx].patient_id),
                            "label": int(yte[j]),
                            "probability": float(prob[j]),
                            "y_pred": int(pred[j]),
                        })

                pd.DataFrame(fold_rows).to_csv(
                    out / "IN_PROGRESS_v17_fold_metrics.csv", index=False,
                    encoding="utf-8-sig"
                )
                pd.DataFrame(pred_rows).to_csv(
                    out / "IN_PROGRESS_v17_patient_predictions.csv", index=False,
                    encoding="utf-8-sig"
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)
    weight_df = pd.DataFrame(weight_rows)
    seed_df = summarize_seed(fold_df, pred_df)
    repeated_df = summarize_repeated(seed_df)

    fold_df.to_csv(out / "v17_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out / "v17_patient_predictions.csv", index=False, encoding="utf-8-sig")
    seed_df.to_csv(out / "v17_seed_summary.csv", index=False, encoding="utf-8-sig")
    repeated_df.to_csv(out / "v17_repeated_cv_summary.csv", index=False, encoding="utf-8-sig")
    weight_df.to_csv(out / "v17_feature_weights.csv", index=False, encoding="utf-8-sig")

    if not weight_df.empty:
        stab = (
            weight_df.assign(
                abs_weight=lambda d: np.abs(d.effective_weight_scaled_space),
                positive=lambda d: (d.effective_weight_scaled_space > 0).astype(float),
            )
            .groupby(["mode", "model", "feature"], as_index=False)
            .agg(
                mean_abs_weight=("abs_weight", "mean"),
                median_abs_weight=("abs_weight", "median"),
                positive_fraction=("positive", "mean"),
                n_outer_fits=("feature", "size"),
            )
            .sort_values(["mode", "model", "mean_abs_weight"], ascending=[True, True, False])
        )
        stab.to_csv(
            out / "v17_feature_weight_stability.csv", index=False, encoding="utf-8-sig"
        )

    comp = compare_reference(seed_df, args.baseline_seed_summary.expanduser().resolve())
    if not comp.empty:
        comp.to_csv(
            out / "v17_vs_mainstream_same_seed.csv", index=False, encoding="utf-8-sig"
        )

    print("\n=== V17 REPEATED-CV SUMMARY ===")
    cols = [
        "mode", "model", "n_seeds", "mean_outer_auc_mean", "mean_outer_auc_sd",
        "balanced_accuracy_fold_mean_mean", "macro_f1_fold_mean_mean",
        "accuracy_fold_mean_mean", "sensitivity_pooled_mean", "specificity_pooled_mean",
    ]
    print(repeated_df[cols].round(4).to_string(index=False))

    if not comp.empty:
        print("\n=== SAME-SEED DELTA VS ORIGINAL MAINSTREAM BASELINE ===")
        cols2 = [
            "seed", "mode", "model", "mean_outer_auc_v17",
            "mean_outer_auc_reference", "delta_auc_v17_minus_reference",
            "delta_ba_v17_minus_reference", "delta_accuracy_v17_minus_reference",
        ]
        print(comp[cols2].round(4).to_string(index=False))

    print(f"\nResults: {out}")


def self_test():
    rng = np.random.default_rng(123)
    channels = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
        "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
    ]
    names = []
    for b in BANDS:
        for ch in channels:
            names.append(f"absdb_{b}_{ch}")
    for b in BANDS:
        for ch in channels:
            names.append(f"rel_{b}_{ch}")

    y = np.array([0] * 20 + [1] * 40, dtype=int)
    X = rng.normal(size=(60, 190))
    X[y == 1, 38:48] += 0.6
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "label", y)
    df.insert(0, "patient_id", [f"P{i:03d}" for i in range(60)])

    cols = validate_table(df)
    dims = {m: len(mode_columns(cols, m)) for m in LOCKED_MODES}
    expected = {
        "base190": 190, "abs95": 95, "rel95": 95, "alpha38": 38,
        "theta_alpha76": 76, "alpha_beta76": 76,
        "pca90": 190, "pca95": 190, "pca_tuned": 190,
    }
    assert dims == expected, (dims, expected)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    pipe, grid = make_pipeline(MODEL_SVM, "pca95", 1)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
    gs.fit(X, y)
    ncomp, ev = pca_info(gs.best_estimator_)
    assert ncomp >= 1 and 0.90 <= ev <= 1.000001
    print(f"SELF_TEST_OK modes={len(LOCKED_MODES)} pca95_components={ncomp}")


def parser():
    p = argparse.ArgumentParser(description="V17 structured spectral ablation + PCA")
    p.add_argument(
        "--feature-csv", type=Path,
        default=Path(
            "/home/thinker/EEGNet/training_results/mainstream_eeg_baselines_5seeds/"
            "patient_bandpower_features.csv"
        ),
    )
    p.add_argument(
        "--baseline-seed-summary", type=Path,
        default=Path(
            "/home/thinker/EEGNet/training_results/mainstream_eeg_baselines_5seeds/"
            "mainstream_seed_summary.csv"
        ),
    )
    p.add_argument(
        "--output-root", type=Path,
        default=Path(
            "/home/thinker/EEGNet/training_results/v17_structured_pca_5seeds"
        ),
    )
    p.add_argument("--seeds", default="92,102,112,122,132")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=4)
    p.add_argument("--modes", default="all")
    p.add_argument("--models", default=f"{MODEL_ENET},{MODEL_SVM}")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--self-test", action="store_true")
    return p


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
