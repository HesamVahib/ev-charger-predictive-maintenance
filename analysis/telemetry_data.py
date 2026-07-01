"""Data loading, leakage control, and feature preparation for charger telemetry.

This file is intentionally boring: read the exported telemetry, keep only light
sanity checks, build rolling features, and split by session. Most of the project
quality depends on not leaking direct alarm/status fields into the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


RANDOM_STATE = 42

DEFAULT_TARGET = "label_future_anomaly_10min"

LABEL_COLUMNS = [
    "label_current_anomaly",
    "label_future_anomaly_5min",
    "label_future_anomaly_10min",
    "label_future_anomaly_15min",
]

ID_TIME_COLUMNS = [
    "sample_id",
    "txId",
    "sessionId",
    "stationId",
    "chargerId",
    "locationUid",
    "vehicleMacAddress",
    "vehicleRunId",
    "sampleTime",
    "timestamp",
    "startTime",
]

# These columns are useful for label creation/rules/reporting but should not be
# used as ordinary ML features because they can reveal the target directly.
DIRECT_LEAKAGE_COLUMNS = [
    "scenario",
    "label_source",
    "anomaly_start_relative_s",
    "anomaly_end_relative_s",
    "status",
    "chargingState",
    "stopReason",
    "unavailableReason",
    "insulationTestStatus",
    "insulationTestFailed",
    "hwFailure",
    "hwFailureReasons_count",
    "alarms_count",
    "errors_count",
    "commNotReady",
    "commNotReadyReasons_count",
    "evCommunicationActive",
    "cableConnected",
    "cablesLocked",
    "rule_score",
]

# Columns for rolling feature creation. Only existing numeric columns are used.
ROLLING_BASE_COLUMNS = [
    "voltage",
    "current",
    "powerW",
    "powerKw",
    "soc",
    "stateOfCharge",
    "currentDemand",
    "targetCurrent",
    "targetVoltage",
    "efficiency",
    "boardTemp",
    "rtTemp",
    "pinTemp",
    "pin2temp",
    "AFEHeatsinkTemp",
    "DCDCHeatsinkTemp",
    "environmentalTemperature",
    "current_error_abs",
    "current_error_ratio",
    "target_current_error",
    "target_voltage_error",
    "power_vi_error_w",
    "temperature_to_ambient_delta",
    "pmc_current_gap",
    "pmc_powerW_gap",
    "pmc_voltage_gap",
    "pmc_temp_gap",
    "pmc_voltageTHD_mean",
    "pmc_voltageTHD_max",
    "missing_ratio",
    "timestamp_gap_seconds",
    "sequence_gap",
]

SAFE_CATEGORICAL_COLUMNS = [
    "chargingProtocol",
    "plugType",
    "plugTypeDetail",
    "voltageCategory",
    "startReason",
    # station/charger can be useful for fleet risk but can reduce transferability.
    # They are disabled by default unless --include-asset-id-features is used.
    "stationId",
    "chargerId",
]


# Small helpers. Kept here because both data prep and the runner use them.

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_bool_to_int(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        lowered = s.astype(str).str.lower()
        if lowered.isin(["true", "false", "nan", "none", "", "0", "1"]).mean() > 0.90:
            return lowered.map({"true": 1, "1": 1, "false": 0, "0": 0}).fillna(0).astype(int)
    return s


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def dataframe_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """Return markdown when tabulate is installed; otherwise fall back to plain text."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the flat telemetry CSV and parse the few time columns we need."""
    df = pd.read_csv(path, low_memory=False)

    for col in ["sampleTime", "timestamp", "startTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Convert boolean-like columns to integers where appropriate.
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup. Keep anomaly patterns; only blank out impossible sensor values."""
    df = df.copy()

    sort_cols = [c for c in ["sessionId", "sampleTime", "timestamp", "sequenceNumber"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Many exported telemetry fields arrive as strings. Convert only columns that are clearly numeric.
    for col in df.columns:
        if col in ID_TIME_COLUMNS:
            continue
        if df[col].dtype == object:
            numeric_candidate = pd.to_numeric(df[col], errors="coerce")
            if numeric_candidate.notna().mean() > 0.80:
                df[col] = numeric_candidate

    for col in df.columns:
        df[col] = safe_bool_to_int(df[col])

    # Do not drop rows here. A bad sensor value may itself be part of the behaviour we want to see.
    bounded = {
        "soc": (0, 100),
        "stateOfCharge": (0, 100),
        "voltage": (0, 1200),
        "current": (-50, 1000),
        "powerW": (-1000, 500000),
        "efficiency": (0, 1.2),
        "boardTemp": (-50, 140),
        "rtTemp": (-50, 140),
        "AFEHeatsinkTemp": (-50, 160),
        "DCDCHeatsinkTemp": (-50, 160),
        "pinTemp": (-50, 160),
        "pin2temp": (-50, 160),
    }
    for col, (lo, hi) in bounded.items():
        if col in df.columns:
            s = as_numeric(df[col])
            df.loc[(s < lo) | (s > hi), col] = np.nan

    return df


def anomaly_rate_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Write the rate tables used for sanity checks before modelling."""
    ensure_dir(output_dir)
    labels = [c for c in LABEL_COLUMNS if c in df.columns]

    rows = []
    for label in labels:
        rows.append({
            "label": label,
            "rows": len(df),
            "positive_rows": int(df[label].sum()),
            "positive_rate": float(df[label].mean()),
        })
    pd.DataFrame(rows).to_csv(output_dir / "overall_anomaly_rates.csv", index=False)

    group_cols = ["scenario", "stationId", "chargerId", "status", "chargingState", "evseId", "connectorId"]
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        agg_dict = {label: ["sum", "mean"] for label in labels}
        agg_dict["sample_id"] = "count" if "sample_id" in df.columns else "size"
        grouped = df.groupby(group_col, dropna=False).agg(agg_dict)
        grouped.columns = ["_".join([str(x) for x in col if x]) for col in grouped.columns]
        grouped = grouped.reset_index().rename(columns={"sample_id_count": "rows", "sample_id_size": "rows"})
        grouped.to_csv(output_dir / f"rates_by_{group_col}.csv", index=False)

    # Session-level risk table.
    if "sessionId" in df.columns:
        session_aggs = {label: ["sum", "mean"] for label in labels}
        for c in ["stationId", "chargerId", "scenario"]:
            if c in df.columns:
                session_aggs[c] = "first"
        if "sample_id" in df.columns:
            session_aggs["sample_id"] = "count"
        sessions = df.groupby("sessionId").agg(session_aggs)
        sessions.columns = ["_".join([str(x) for x in col if x]) for col in sessions.columns]
        sessions = sessions.reset_index().rename(columns={"sample_id_count": "rows"})
        sort_label = labels[0] + "_mean" if labels else "rows"
        sessions = sessions.sort_values(sort_label, ascending=False)
        sessions.to_csv(output_dir / "rates_by_session.csv", index=False)


def make_rule_score(df: pd.DataFrame) -> pd.Series:
    """Create a simple direct rule-based anomaly score from operational flags.

    This is useful for reporting and operational dashboards. It should not be used
    as an ordinary feature when evaluating ML models against weak labels made from
    the same flags.
    """
    score = pd.Series(0.0, index=df.index)

    def bump(condition: pd.Series, value: float) -> None:
        nonlocal score
        condition = condition.fillna(False).astype(bool)
        score = np.maximum(score, condition.astype(float) * value)

    if "insulationTestStatus" in df.columns:
        bump(df["insulationTestStatus"].astype(str).str.upper().ne("PASSED"), 0.95)
    if "insulationTestFailed" in df.columns:
        bump(as_numeric(df["insulationTestFailed"]).fillna(0).gt(0), 0.95)
    if "hwFailure" in df.columns:
        bump(as_numeric(df["hwFailure"]).fillna(0).gt(0), 0.95)
    if "alarms_count" in df.columns:
        bump(as_numeric(df["alarms_count"]).fillna(0).gt(0), 0.85)
    if "errors_count" in df.columns:
        bump(as_numeric(df["errors_count"]).fillna(0).gt(0), 0.85)
    if "unavailableReason" in df.columns:
        bump(df["unavailableReason"].notna() & df["unavailableReason"].astype(str).ne(""), 0.85)
    if "commNotReady" in df.columns:
        bump(as_numeric(df["commNotReady"]).fillna(0).gt(0), 0.70)
    if {"evCommunicationActive", "isCharging"}.issubset(df.columns):
        bump(as_numeric(df["isCharging"]).fillna(0).gt(0) & as_numeric(df["evCommunicationActive"]).fillna(1).eq(0), 0.70)
    if {"cableConnected", "active"}.issubset(df.columns):
        bump(as_numeric(df["active"]).fillna(0).gt(0) & as_numeric(df["cableConnected"]).fillna(1).eq(0), 0.75)

    temp_cols = [c for c in ["boardTemp", "rtTemp", "pinTemp", "pin2temp", "AFEHeatsinkTemp", "DCDCHeatsinkTemp"] if c in df]
    if temp_cols:
        max_temp = df[temp_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
        bump(max_temp.gt(75), 0.60)
        bump(max_temp.gt(90), 0.90)

    if {"currentDemand", "current"}.issubset(df.columns):
        demand = as_numeric(df["currentDemand"]).fillna(0)
        current = as_numeric(df["current"]).fillna(0)
        bump(demand.gt(30) & current.lt(5), 0.80)

    if "power_vi_error_w" in df.columns:
        err = as_numeric(df["power_vi_error_w"]).abs()
        bump(err.gt(5000), 0.55)
        bump(err.gt(15000), 0.80)

    return pd.Series(score, index=df.index, name="rule_score")


def add_rolling_features(df: pd.DataFrame, window_samples: int = 6, sample_seconds: int = 10) -> pd.DataFrame:
    """Add past-window rolling mean/std/slope features per session.

    With 10-second sampling, window_samples=6 corresponds to roughly 60 seconds.
    This vectorized implementation is much faster than per-column rolling lambdas.
    """
    df = df.copy()
    if "sessionId" not in df.columns:
        raise ValueError("sessionId column is required for rolling features.")

    # 6 samples = 60s in the default run. This compact set came from the first notebook pass;
    # adding every raw field made the plots noisier and did not help the first prototype.
    preferred = [
        "voltage", "current", "powerW", "soc", "currentDemand",
        "efficiency", "boardTemp", "rtTemp", "AFEHeatsinkTemp",
        "DCDCHeatsinkTemp", "current_error_abs", "power_vi_error_w",
        "pmc_current_gap", "pmc_powerW_gap", "pmc_temp_gap", "missing_ratio",
    ]
    existing = [c for c in preferred if c in df.columns]
    if not existing:
        return df

    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group = df.groupby("sessionId", sort=False)[existing]
    roll_mean = group.rolling(window_samples, min_periods=2).mean().reset_index(level=0, drop=True)
    roll_std = group.rolling(window_samples, min_periods=2).std().reset_index(level=0, drop=True)
    slopes = group.diff(window_samples - 1) / max((window_samples - 1) * sample_seconds, 1)

    roll_mean = roll_mean.add_suffix(f"_roll_mean_{window_samples}")
    roll_std = roll_std.add_suffix(f"_roll_std_{window_samples}")
    slopes = slopes.add_suffix(f"_slope_{window_samples}")

    return pd.concat([df, roll_mean, roll_std, slopes], axis=1).copy()


def select_feature_columns(
    df: pd.DataFrame,
    target: str,
    include_asset_id_features: bool = False,
) -> Tuple[List[str], List[str]]:
    """Select safe numeric and categorical feature columns."""
    exclude = set(ID_TIME_COLUMNS)
    exclude.update(LABEL_COLUMNS)
    exclude.update(DIRECT_LEAKAGE_COLUMNS)
    exclude.add(target)

    # Always exclude exact future/current label fields and anomaly metadata.
    exclude.update([c for c in df.columns if c.startswith("label_")])

    if not include_asset_id_features:
        exclude.update(["stationId", "chargerId"])

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
        elif col in SAFE_CATEGORICAL_COLUMNS and col not in exclude:
            # Avoid very high-cardinality categoricals.
            if df[col].nunique(dropna=True) <= 50:
                categorical_cols.append(col)

    # Remove constant features.
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    categorical_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) > 1]
    return numeric_cols, categorical_cols


def chronological_session_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> Dict[str, pd.Series]:
    """Chronological session split. Row-level splits would leak session behaviour."""
    if "sessionId" not in df.columns:
        raise ValueError("sessionId is required for session-based split.")

    if "sampleTime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["sampleTime"]):
        session_start = df.groupby("sessionId")["sampleTime"].min().sort_values()
    elif "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        session_start = df.groupby("sessionId")["timestamp"].min().sort_values()
    else:
        session_start = pd.Series(range(df["sessionId"].nunique()), index=df["sessionId"].drop_duplicates())

    sessions = session_start.index.to_numpy()
    n = len(sessions)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_sessions = set(sessions[:n_train])
    val_sessions = set(sessions[n_train:n_train + n_val])
    test_sessions = set(sessions[n_train + n_val:])

    return {
        "train": df["sessionId"].isin(train_sessions),
        "val": df["sessionId"].isin(val_sessions),
        "test": df["sessionId"].isin(test_sessions),
    }
