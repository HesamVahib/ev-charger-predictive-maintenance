"""Models, thresholds, explanations, and plots for the anomaly pipeline.

The split is not meant to be a polished package. It is just the part of the
pipeline that changes fastest while experimenting with models and alert logic.
"""

from __future__ import annotations

import time
from pathlib import Path
from tkinter.font import names
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
from supabase_auth import model
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telemetry_data import RANDOM_STATE, as_numeric, ensure_dir

FEATURE_LABELS = {
    "pmc_temp_gap_roll_mean_6": "60-s average module temperature imbalance",
    "pmc_temp_gap": "Module temperature imbalance",
    "temperature_to_ambient_delta": "Temperature rise above ambient",
    "voltage_roll_std_6": "60-s voltage instability",
    "pmc_current_gap_roll_mean_6": "60-s average module current imbalance",
    "pmc_current_gap_roll_std_6": "Variation in module current imbalance",
    "pmc_powerW_gap_roll_mean_6": "60-s average module power imbalance",
    "current_error": "Current tracking error",
    "pmc_powerW_gap_roll_std_6": "Variation in module power imbalance",
    "pmc_temp_gap_roll_std_6": "Variation in module temperature imbalance",
    "missing_ratio_roll_std_6": "Variation in missing telemetry rate",
    "missing_ratio_roll_mean_6": "60-s average missing telemetry rate",
    "efficiency": "Charging efficiency",
    "efficiency_roll_mean_6": "60-s average charging efficiency",
    "boardTemp_roll_std_6": "Variation in board temperature",
    "voltage_slope_6": "Voltage change rate",
    "pmc_current_gap": "Module current imbalance",
    "relativeTsSeconds": "Time elapsed in charging session",
    "current_error_ratio": "Relative current tracking error",
    "pmc_current_std": "Variation in module current",
    "efficiency_roll_std_6": "Variation in charging efficiency",
    "AFEHeatsinkTemp_roll_std_6": "Variation in AFE heatsink temperature",
    "DCDCHeatsinkTemp_roll_std_6": "Variation in DC-DC heatsink temperature",
    "missing_ratio_slope_6": "Change rate of missing telemetry",
    "pmc_voltage_std": "Variation in module voltage",
    "pmc_voltage_gap": "Module voltage imbalance",
    "pmc_voltage_gap_roll_mean_6": "60-s average module voltage imbalance",
    "pmc_voltage_gap_roll_std_6": "Variation in module voltage imbalance",
    "boardTemp_roll_mean_6": "60-s average board temperature",
    "AFEHeatsinkTemp_roll_mean_6": "60-s average AFE heatsink temperature",
    "DCDCHeatsinkTemp_roll_mean_6": "60-s average DC-DC heatsink temperature",
}

MODEL_LABELS = {
    "supervised_weighted_ensemble": "Supervised ensemble",
    "random_forest": "Random Forest",
    "ensemble": "Final ensemble",
    "logistic_regression": "Logistic Regression",
    "extra_trees": "Extra Trees",
    "isolation_forest_unsupervised": "Isolation Forest",
}

def readable_model_name(model_name):
    return MODEL_LABELS.get(model_name, model_name)

def readable_model_name(model_name):
    return MODEL_LABELS.get(model_name, model_name)

def readable_feature_name(feature_name):
    """Use clean engineering labels in plots, but keep raw names in data files."""
    return FEATURE_LABELS.get(feature_name, feature_name)

def get_one_hot_encoder() -> OneHotEncoder:
    # This project was run with a recent scikit-learn build. No version shim here.
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def get_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a sklearn pipeline."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X)
        return normalize_scores(scores)
    raise ValueError("Pipeline does not support probability or decision scores.")


def normalize_scores(scores: np.ndarray, lower: Optional[float] = None, upper: Optional[float] = None) -> np.ndarray:
    """Normalize scores to 0-1 using robust min/max bounds."""
    scores = np.asarray(scores, dtype=float)
    if lower is None:
        lower = np.nanpercentile(scores, 1)
    if upper is None:
        upper = np.nanpercentile(scores, 99)
    denom = max(upper - lower, 1e-9)
    return np.clip((scores - lower) / denom, 0.0, 1.0)


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", get_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_supervised_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: Path,
    fast: bool = False,
) -> List[Dict[str, object]]:
    """Fit the small supervised model set used in the report."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=300,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=40 if fast else 120,
            max_depth=10 if fast else 14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=40 if fast else 120,
            max_depth=12 if fast else 18,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # HistGradientBoosting can be added later, but the default script keeps
        # the MVP fast and reproducible on ordinary laptops.
    }

    results: List[Dict[str, object]] = []
    ensure_dir(output_dir / "models")

    for name, clf in models.items():
        print(f"Training supervised model: {name}")
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", clf),
            ]
        )
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - t0
        val_proba = get_probability(pipe, X_val)

        val_pr = average_precision_score(y_val, val_proba) if y_val.nunique() == 2 else np.nan
        val_roc = roc_auc_score(y_val, val_proba) if y_val.nunique() == 2 else np.nan

        joblib.dump(pipe, output_dir / "models" / f"{name}.joblib")
        results.append({
            "name": name,
            "pipeline": pipe,
            "val_proba": val_proba,
            "fit_seconds": fit_seconds,
            "val_pr_auc": float(val_pr),
            "val_roc_auc": float(val_roc),
        })
        print(f"  done in {fit_seconds:.1f}s | val PR-AUC={val_pr:.4f} ROC-AUC={val_roc:.4f}")

    return results


def train_isolation_forest(
    X_train_normal: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: Path,
) -> Tuple[Pipeline, np.ndarray]:
    """Isolation Forest baseline; trained only on normal rows."""
    print("Training unsupervised model: isolation_forest on normal training rows")
    pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", IsolationForest(
                n_estimators=200,
                contamination="auto",  # TODO: retune when real failure logs are available.
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )
    pipe.fit(X_train_normal)
    # Higher score should mean more abnormal.
    val_raw = -pipe.decision_function(X_val)
    joblib.dump(pipe, output_dir / "models" / "isolation_forest_unsupervised.joblib")
    return pipe, val_raw


def choose_threshold(y_val: pd.Series, score_val: np.ndarray) -> float:
    """Balanced threshold = best validation F1."""
    precision, recall, thresholds = precision_recall_curve(y_val, score_val)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def choose_threshold_by_false_alarm(
    y_val: pd.Series,
    score_val: np.ndarray,
    max_false_alarm_rate: float = 0.01,
) -> float:
    """Choose a conservative threshold with validation false-alarm rate below a target.

    Among thresholds satisfying the false-alarm constraint, select the one with the
    highest F1 score. If no candidate satisfies the constraint, fall back to the
    balanced F1 threshold.
    """
    y = np.asarray(y_val, dtype=int)
    s = np.asarray(score_val, dtype=float)
    if len(s) == 0 or len(np.unique(y)) < 2:
        return 0.5

    # Candidate thresholds from score quantiles keep computation stable and fast.
    quantiles = np.linspace(0.0, 1.0, 501)
    candidates = np.unique(np.nanquantile(s, quantiles))
    rows = []
    for thr in candidates:
        pred = (s >= thr).astype(int)
        normal_count = max((y == 0).sum(), 1)
        fpr = ((pred == 1) & (y == 0)).sum() / normal_count
        if fpr <= max_false_alarm_rate:
            rows.append((
                f1_score(y, pred, zero_division=0),
                recall_score(y, pred, zero_division=0),
                precision_score(y, pred, zero_division=0),
                -fpr,
                float(thr),
            ))
    if not rows:
        return choose_threshold(y_val, score_val)
    # Maximize F1, then recall, then precision, then lower false alarm rate.
    return max(rows)[-1]


def critical_threshold_from_validation(
    val_scores: np.ndarray,
    warning_threshold: float,
    percentile: float = 95.0,
) -> float:
    """Choose a critical threshold above the warning threshold.

    The critical threshold is set from high-risk validation scores. This keeps
    Normal/Warning/Critical aligned with the model's calibrated warning threshold.
    """
    scores = np.asarray(val_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float(min(max(warning_threshold + 0.15, 0.0), 1.0))
    high_scores = scores[scores >= warning_threshold]
    if high_scores.size >= 10:
        crit = float(np.nanpercentile(high_scores, percentile))
    else:
        crit = float(np.nanpercentile(scores, percentile))
    crit = max(crit, warning_threshold + 1e-6)
    return float(np.clip(crit, 0.0, 1.0))


def calibrated_health_scores(
    anomaly_scores: np.ndarray,
    warning_threshold: float,
    critical_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert anomaly scores to calibrated health scores and status labels.

    Calibration rule:
    - Below warning threshold: Normal, health 80-100.
    - Between warning and critical: Warning, health 50-80.
    - Above critical threshold: Critical, health 0-50.
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    warning_threshold = float(warning_threshold)
    critical_threshold = float(max(critical_threshold, warning_threshold + 1e-6))
    health = np.zeros_like(scores, dtype=float)
    status = np.empty(scores.shape, dtype=object)

    normal_mask = scores < warning_threshold
    warning_mask = (scores >= warning_threshold) & (scores < critical_threshold)
    critical_mask = scores >= critical_threshold

    if warning_threshold > 1e-9:
        normal_ratio = np.clip(scores[normal_mask] / warning_threshold, 0.0, 1.0)
        health[normal_mask] = 100.0 - 20.0 * normal_ratio
    else:
        health[normal_mask] = 100.0
    status[normal_mask] = "Normal"

    denom_warn = max(critical_threshold - warning_threshold, 1e-9)
    warning_ratio = np.clip((scores[warning_mask] - warning_threshold) / denom_warn, 0.0, 1.0)
    health[warning_mask] = 80.0 - 30.0 * warning_ratio
    status[warning_mask] = "Warning"

    denom_crit = max(1.0 - critical_threshold, 1e-9)
    critical_ratio = np.clip((scores[critical_mask] - critical_threshold) / denom_crit, 0.0, 1.0)
    health[critical_mask] = 50.0 - 50.0 * critical_ratio
    status[critical_mask] = "Critical"

    return np.clip(health, 0.0, 100.0), status


def weighted_average_scores(scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted average helper for the ensemble scores."""
    keys = list(scores.keys())
    w = np.array([max(weights.get(k, 0.0), 0.0) for k in keys], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    matrix = np.vstack([scores[k] for k in keys])
    return np.average(matrix, axis=0, weights=w)


def evaluate_binary(y_true: pd.Series, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_alarm_rate": float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "missed_anomaly_rate": float(((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
        "alert_rate": float(y_pred.mean()),
        "positive_rate": float(y_true.mean()),
    }
    if y_true.nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def create_prediction_table(
    df_part: pd.DataFrame,
    target: str,
    scores: Dict[str, np.ndarray],
    ensemble_score: np.ndarray,
    warning_threshold: float,
    critical_threshold: float,
    alert_mode: str,
    rule_score: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Create row-level prediction output with calibrated health score and status."""
    meta_cols = [
        "sample_id", "sessionId", "stationId", "chargerId", "sampleTime", "timestamp",
        "relativeTsSeconds", "scenario", target,
    ]
    meta_cols = [c for c in meta_cols if c in df_part.columns]
    out = df_part[meta_cols].copy()

    for name, sc in scores.items():
        out[f"score_{name}"] = sc
    if rule_score is not None:
        out["score_rule"] = rule_score.loc[df_part.index].to_numpy(dtype=float)

    out["alert_mode"] = alert_mode
    out["ensemble_score"] = ensemble_score
    out["warning_threshold"] = float(warning_threshold)
    out["critical_threshold"] = float(critical_threshold)
    out["predicted_anomaly"] = (ensemble_score >= warning_threshold).astype(int)
    health, status = calibrated_health_scores(ensemble_score, warning_threshold, critical_threshold)
    out["health_score"] = np.round(health, 2)
    out["health_status"] = status.astype(str)
    return out


def top_zscore_explanations(
    df_part: pd.DataFrame,
    numeric_cols: List[str],
    train_normal: pd.DataFrame,
    prediction_table: pd.DataFrame,
    max_rows: int = 1000,
) -> pd.DataFrame:
    """Rank top abnormal numeric features by absolute z-score from normal training data."""
    # Prefer features with interpretable names and not too many rolling duplicates.
    cols = [c for c in numeric_cols if c in df_part.columns]
    if not cols:
        return pd.DataFrame()

    normal_mean = train_normal[cols].apply(pd.to_numeric, errors="coerce").mean()
    normal_std = train_normal[cols].apply(pd.to_numeric, errors="coerce").std().replace(0, np.nan).fillna(1.0)

    # Explain highest-risk rows first.
    candidates = prediction_table.sort_values("ensemble_score", ascending=False).head(max_rows)
    rows = []
    for idx in candidates.index:
        values = df_part.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
        z = ((values - normal_mean).abs() / normal_std).replace([np.inf, -np.inf], np.nan).fillna(0)
        top = z.sort_values(ascending=False).head(3)
        base = {
            "sample_index": idx,
            "sessionId": df_part.loc[idx, "sessionId"] if "sessionId" in df_part else None,
            "chargerId": df_part.loc[idx, "chargerId"] if "chargerId" in df_part else None,
            "scenario": df_part.loc[idx, "scenario"] if "scenario" in df_part else None,
            "ensemble_score": float(prediction_table.loc[idx, "ensemble_score"]),
            "health_status": prediction_table.loc[idx, "health_status"],
        }
        for rank, (feature, score) in enumerate(top.items(), start=1):
            base[f"top_{rank}_feature"] = feature
            base[f"top_{rank}_zscore"] = float(score)
        rows.append(base)
    return pd.DataFrame(rows)


def early_warning_summary(predictions: pd.DataFrame, df_test: pd.DataFrame, target: str) -> pd.DataFrame:
    """Estimate early-warning lead time by session when anomaly_start_relative_s exists."""
    if "anomaly_start_relative_s" not in df_test.columns or "relativeTsSeconds" not in df_test.columns:
        return pd.DataFrame()
    if "sessionId" not in df_test.columns:
        return pd.DataFrame()

    tmp = predictions[["sessionId", "predicted_anomaly"]].copy()
    tmp["relativeTsSeconds"] = df_test.loc[predictions.index, "relativeTsSeconds"].values
    tmp["anomaly_start_relative_s"] = df_test.loc[predictions.index, "anomaly_start_relative_s"].values
    tmp[target] = df_test.loc[predictions.index, target].values if target in df_test else np.nan

    rows = []
    for sid, g in tmp.groupby("sessionId"):
        starts = pd.to_numeric(g["anomaly_start_relative_s"], errors="coerce").dropna()
        if starts.empty:
            continue
        anomaly_start = float(starts.min())
        alert_times = pd.to_numeric(g.loc[g["predicted_anomaly"] == 1, "relativeTsSeconds"], errors="coerce").dropna()
        if alert_times.empty:
            rows.append({
                "sessionId": sid,
                "anomaly_start_relative_s": anomaly_start,
                "first_alert_time_s": np.nan,
                "lead_time_s": np.nan,
                "alerted_before_or_at_start": 0,
            })
            continue
        first_alert = float(alert_times.min())
        lead = anomaly_start - first_alert
        rows.append({
            "sessionId": sid,
            "anomaly_start_relative_s": anomaly_start,
            "first_alert_time_s": first_alert,
            "lead_time_s": lead,
            "alerted_before_or_at_start": int(lead >= 0),
        })
    return pd.DataFrame(rows)


def save_model_comparison(
    output_dir: Path,
    target: str,
    y_val: pd.Series,
    y_test: pd.Series,
    val_scores: Dict[str, np.ndarray],
    test_scores: Dict[str, np.ndarray],
    threshold: float,
    fit_times: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for name, sc in val_scores.items():
        thr = choose_threshold(y_val, sc) if y_val.nunique() == 2 else 0.5
        r = evaluate_binary(y_val, sc, thr)
        r.update({"dataset": "validation", "model": name, "target": target, "fit_seconds": fit_times.get(name, np.nan)})
        rows.append(r)
    for name, sc in test_scores.items():
        # Use final ensemble threshold for ensemble, own val threshold for others.
        thr = threshold if name == "ensemble" else choose_threshold(y_val, val_scores[name])
        r = evaluate_binary(y_test, sc, thr)
        r.update({"dataset": "test", "model": name, "target": target, "fit_seconds": fit_times.get(name, np.nan)})
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "model_comparison.csv", index=False)
    return table


def save_feature_importance(pipeline: Pipeline, output_dir: Path) -> None:
    """Save tree feature importance for pipelines with feature_importances_."""
    try:
        pre = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return
        names = pre.get_feature_names_out()
        imp = model.feature_importances_

        table = pd.DataFrame({
                    "feature": names,
                    "feature_label": [readable_feature_name(x) for x in names],
                    "importance": imp,
            }).sort_values("importance", ascending=False)

        table.to_csv(output_dir / "feature_importance_random_forest.csv", index=False)
    except Exception as exc:
        print(f"Could not save feature importance: {exc}")


def evaluate_threshold_modes(
    y_val: pd.Series,
    y_test: pd.Series,
    val_score: np.ndarray,
    test_score: np.ndarray,
    balanced_threshold: float,
    conservative_threshold: float,
    target: str,
    tables_dir: Path,
) -> pd.DataFrame:
    """Save validation/test metrics for balanced and conservative alert modes."""
    rows = []
    for mode, thr in [("balanced", balanced_threshold), ("conservative", conservative_threshold)]:
        r_val = evaluate_binary(y_val, val_score, thr)
        r_val.update({"dataset": "validation", "alert_mode": mode, "target": target})
        rows.append(r_val)
        r_test = evaluate_binary(y_test, test_score, thr)
        r_test.update({"dataset": "test", "alert_mode": mode, "target": target})
        rows.append(r_test)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "threshold_modes.csv", index=False)
    return table


def plot_model_comparison(comparison: pd.DataFrame, figures_dir: Path) -> None:
    """Plot test precision, recall and F1 by model."""
    test = comparison[comparison["dataset"].eq("test")].copy()
    if test.empty:
        return
    test = test.sort_values("f1", ascending=False)
    test["model_label"] = test["model"].map(readable_model_name)
    metrics = ["precision", "recall", "f1"]
    ax = test.set_index("model_label")[metrics].plot(kind="bar", figsize=(11, 5))
    ax.set_title("Model comparison on test set")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_precision_recall_f1.png", dpi=180)
    plt.close()


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, figures_dir: Path, title: str) -> None:
    """Plot confusion matrix for the selected ensemble mode."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_selected_ensemble.png", dpi=180)
    plt.close()


def plot_score_distribution(predictions: pd.DataFrame, target: str, figures_dir: Path) -> None:
    """Plot anomaly-score distribution grouped by true label."""
    if target not in predictions.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    normal_scores = predictions.loc[predictions[target] == 0, "ensemble_score"].dropna()
    anomaly_scores = predictions.loc[predictions[target] == 1, "ensemble_score"].dropna()
    ax.hist(normal_scores, bins=40, alpha=0.65, label="True normal")
    ax.hist(anomaly_scores, bins=40, alpha=0.65, label="True anomaly")
    if "warning_threshold" in predictions.columns:
        ax.axvline(float(predictions["warning_threshold"].iloc[0]), linestyle="--", label="Warning threshold")
    if "critical_threshold" in predictions.columns:
        ax.axvline(float(predictions["critical_threshold"].iloc[0]), linestyle=":", label="Critical threshold")
    ax.set_title("Ensemble anomaly-score distribution")
    ax.set_xlabel("Ensemble anomaly score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_score_distribution.png", dpi=180)
    plt.close()


def plot_health_timeline(predictions: pd.DataFrame, target: str, figures_dir: Path) -> None:
    """Plot health-score and anomaly-score timelines for a representative high-risk session."""
    if "sessionId" not in predictions.columns:
        return

    session_scores = predictions.groupby("sessionId")["ensemble_score"].max().sort_values(ascending=False)
    if session_scores.empty:
        return

    sid = session_scores.index[0]
    g = predictions[predictions["sessionId"].eq(sid)].copy()

    if "relativeTsSeconds" in g.columns:
        g = g.sort_values("relativeTsSeconds")
        x = g["relativeTsSeconds"] / 60.0
        xlabel = "Session time (minutes)"
    elif "sampleTime" in g.columns:
        g = g.sort_values("sampleTime")
        x = pd.to_datetime(g["sampleTime"], errors="coerce")
        xlabel = "Time"
    else:
        g = g.reset_index(drop=True)
        x = g.index
        xlabel = "Sample index"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, g["health_score"], marker="o", markersize=2, linewidth=1)
    ax.axhline(80, linestyle="--", label="Normal/Warning boundary")
    ax.axhline(50, linestyle=":", label="Warning/Critical boundary")
    ax.set_title("Health-score timeline for a representative high-risk session")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Health score")
    ax.set_ylim(-2, 102)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "health_score_timeline_representative_session.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, g["ensemble_score"], marker="o", markersize=2, linewidth=1)
    if "warning_threshold" in g.columns:
        ax.axhline(float(g["warning_threshold"].iloc[0]), linestyle="--", label="Warning threshold")
    if "critical_threshold" in g.columns:
        ax.axhline(float(g["critical_threshold"].iloc[0]), linestyle=":", label="Critical threshold")
    ax.set_title("Anomaly-score timeline for a representative high-risk session")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ensemble anomaly score")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "anomaly_score_timeline_representative_session.png", dpi=180)
    plt.close()


def plot_explanation_counts(explanations, figures_dir):
    if explanations.empty or "top_1_feature" not in explanations.columns:
        return

    plot_col = "top_1_feature_label" if "top_1_feature_label" in explanations.columns else "top_1_feature"
    counts = explanations[plot_col].value_counts().head(15)

    if counts.empty:
        return

    ax = counts.sort_values().plot(kind="barh", figsize=(10, 6))
    ax.set_title("Most Frequent Abnormal Signals")
    ax.set_xlabel("Count among high-risk samples")
    ax.set_ylabel("Signal")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_abnormal_feature_counts.png", dpi=180)
    plt.close()


def plot_feature_importance_if_available(tables_dir: Path, figures_dir: Path) -> None:
    """Plot Random Forest feature importance if the table exists."""
    path = tables_dir / "feature_importance_random_forest.csv"
    if not path.exists():
        return
    imp = pd.read_csv(path).head(20)
    if imp.empty:
        return

    if "feature_label" in imp.columns:
        labels = imp["feature_label"]
    else:
        labels = imp["feature"].map(readable_feature_name)

    ax = pd.Series(imp["importance"].values, index=labels).sort_values().plot(
        kind="barh", figsize=(10, 6)
    )
    ax.set_title("Most Important Signals Used by the Random Forest Model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Signal")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_random_forest.png", dpi=180)
    plt.close()


def generate_result_plots(
    figures_dir: Path,
    tables_dir: Path,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    explanations: pd.DataFrame,
    target: str,
) -> None:
    """Generate key figures for report/GitHub/application use."""
    ensure_dir(figures_dir)
    plot_model_comparison(comparison, figures_dir)
    if target in predictions.columns and "predicted_anomaly" in predictions.columns:
        plot_confusion(
            predictions[target].astype(int),
            predictions["predicted_anomaly"].astype(int).to_numpy(),
            figures_dir,
            title="Selected ensemble confusion matrix",
        )
    plot_score_distribution(predictions, target, figures_dir)
    plot_health_timeline(predictions, target, figures_dir)
    plot_explanation_counts(explanations, figures_dir)
    plot_feature_importance_if_available(tables_dir, figures_dir)
