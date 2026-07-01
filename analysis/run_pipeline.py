"""Command-line runner for the EV charger anomaly pipeline.

Keep this file short enough to read top-to-bottom. Data prep lives in
``telemetry_data.py`` and model/report code lives in ``anomaly_models.py``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from telemetry_data import (
    DEFAULT_TARGET,
    LABEL_COLUMNS,
    add_rolling_features,
    anomaly_rate_summary,
    basic_cleaning,
    chronological_session_split,
    dataframe_to_markdown_safe,
    ensure_dir,
    load_data,
    make_rule_score,
    select_feature_columns,
)

from anomaly_models import (
    choose_threshold,
    choose_threshold_by_false_alarm,
    create_prediction_table,
    critical_threshold_from_validation,
    early_warning_summary,
    evaluate_threshold_modes,
    generate_result_plots,
    get_probability,
    normalize_scores,
    readable_feature_name,
    save_feature_importance,
    save_model_comparison,
    top_zscore_explanations,
    train_isolation_forest,
    train_supervised_models,
    weighted_average_scores,
)

warnings.filterwarnings("ignore", category=UserWarning)

def run_pipeline(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output_dir = ensure_dir(args.output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    models_dir = ensure_dir(output_dir / "models")
    figures_dir = ensure_dir(output_dir / "figures")

    print(f"Loading data: {data_path}")
    df = load_data(data_path)
    df = basic_cleaning(df)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available label columns: {[c for c in LABEL_COLUMNS if c in df.columns]}")

    print("Assessing anomaly rates...")
    anomaly_rate_summary(df, tables_dir)

    print("Creating rule-score report...")
    rule_score = make_rule_score(df)
    df["rule_score"] = rule_score
    # Defragment after adding rule score; rolling features add many columns later.
    df = df.copy()

    print("Adding rolling features...")
    df = add_rolling_features(df, window_samples=args.window_samples, sample_seconds=args.sample_seconds)

    print("Selecting features...")
    numeric_cols, categorical_cols = select_feature_columns(
        df,
        target=args.target,
        include_asset_id_features=args.include_asset_id_features,
    )
    print(f"Selected numeric features: {len(numeric_cols)}")
    print(f"Selected categorical features: {categorical_cols}")

    # Save feature list for reproducibility.
    with open(output_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, f, indent=2)

    # Keep only rows with known target.
    df_model = df[df[args.target].notna()].copy()
    df_model[args.target] = pd.to_numeric(df_model[args.target], errors="coerce").fillna(0).astype(int)

    masks = chronological_session_split(df_model)
    train_df = df_model[masks["train"]].copy()
    val_df = df_model[masks["val"]].copy()
    test_df = df_model[masks["test"]].copy()

    split_summary = pd.DataFrame([
        {"split": "train", "rows": len(train_df), "sessions": train_df["sessionId"].nunique(), "positive_rate": train_df[args.target].mean()},
        {"split": "validation", "rows": len(val_df), "sessions": val_df["sessionId"].nunique(), "positive_rate": val_df[args.target].mean()},
        {"split": "test", "rows": len(test_df), "sessions": test_df["sessionId"].nunique(), "positive_rate": test_df[args.target].mean()},
    ])
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)
    print(split_summary)

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df[args.target]
    X_val = val_df[numeric_cols + categorical_cols]
    y_val = val_df[args.target]
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df[args.target]

    if y_train.nunique() < 2:
        raise ValueError("Training target has only one class. Adjust split or target.")
    if y_val.nunique() < 2:
        print("Warning: validation target has only one class. Metrics/threshold may be unstable.")
    if y_test.nunique() < 2:
        print("Warning: test target has only one class. Metrics may be unstable.")

    print("Training supervised ensemble models...")
    model_results = train_supervised_models(
        X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, output_dir, fast=args.fast
    )

    val_scores: Dict[str, np.ndarray] = {}
    test_scores: Dict[str, np.ndarray] = {}
    fit_times: Dict[str, float] = {}

    for res in model_results:
        val_scores[res["name"]] = res["val_proba"]
        test_scores[res["name"]] = get_probability(res["pipeline"], X_test)
        fit_times[res["name"]] = res["fit_seconds"]

    # Unsupervised model trained only on normal training rows.
    normal_train_df = train_df[train_df[args.target] == 0]
    if len(normal_train_df) > 50:
        iforest_pipe, iforest_val_raw = train_isolation_forest(
            normal_train_df[numeric_cols + categorical_cols], X_val, numeric_cols, categorical_cols, output_dir
        )
        iforest_test_raw = -iforest_pipe.decision_function(X_test)
        lower = np.nanpercentile(iforest_val_raw, 1)
        upper = np.nanpercentile(iforest_val_raw, 99)
        val_scores["isolation_forest_unsupervised"] = normalize_scores(iforest_val_raw, lower, upper)
        test_scores["isolation_forest_unsupervised"] = normalize_scores(iforest_test_raw, lower, upper)
        fit_times["isolation_forest_unsupervised"] = np.nan
    else:
        print("Skipping Isolation Forest: not enough normal training rows.")

    # Weight supervised models by validation PR-AUC. Simple, transparent, and good enough here.
    supervised_names = [r["name"] for r in model_results]
    weights = {}
    for name in supervised_names:
        if y_val.nunique() == 2:
            weights[name] = max(average_precision_score(y_val, val_scores[name]), 1e-6)
        else:
            weights[name] = 1.0

    val_supervised_ensemble = weighted_average_scores({n: val_scores[n] for n in supervised_names}, weights)
    test_supervised_ensemble = weighted_average_scores({n: test_scores[n] for n in supervised_names}, weights)
    val_scores["supervised_weighted_ensemble"] = val_supervised_ensemble
    test_scores["supervised_weighted_ensemble"] = test_supervised_ensemble

    # The unsupervised score is kept as a small stability layer, not as the main detector.
    if "isolation_forest_unsupervised" in val_scores:
        val_final = 0.85 * val_supervised_ensemble + 0.15 * val_scores["isolation_forest_unsupervised"]
        test_final = 0.85 * test_supervised_ensemble + 0.15 * test_scores["isolation_forest_unsupervised"]
    else:
        val_final = val_supervised_ensemble
        test_final = test_supervised_ensemble
    val_scores["ensemble"] = val_final
    test_scores["ensemble"] = test_final

    balanced_threshold = choose_threshold(y_val, val_final) if y_val.nunique() == 2 else 0.5
    conservative_threshold = choose_threshold_by_false_alarm(
        y_val, val_final, max_false_alarm_rate=args.conservative_fpr
    ) if y_val.nunique() == 2 else 0.5

    main_alert_mode = "balanced" if args.alert_mode == "both" else args.alert_mode
    threshold_map = {
        "balanced": balanced_threshold,
        "conservative": conservative_threshold,
    }
    threshold = threshold_map[main_alert_mode]
    critical_threshold = critical_threshold_from_validation(
        val_final, threshold, percentile=args.critical_percentile
    )
    print(f"Balanced threshold from validation: {balanced_threshold:.4f}")
    print(f"Conservative threshold from validation: {conservative_threshold:.4f} "
          f"(target validation FPR <= {args.conservative_fpr:.3f})")
    print(f"Selected alert mode: {main_alert_mode}")
    print(f"Selected warning threshold: {threshold:.4f}")
    print(f"Selected critical threshold: {critical_threshold:.4f}")

    threshold_modes = evaluate_threshold_modes(
        y_val, y_test, val_final, test_final,
        balanced_threshold, conservative_threshold, args.target, tables_dir
    )
    print("Threshold-mode comparison:")
    print(threshold_modes[threshold_modes["dataset"].eq("test")])

    print("Evaluating models...")
    comparison = save_model_comparison(
        tables_dir, args.target, y_val, y_test, val_scores, test_scores, threshold, fit_times
    )
    print(comparison[comparison["dataset"].eq("test")].sort_values("f1", ascending=False).head(10))

    predictions = create_prediction_table(
        test_df,
        args.target,
        scores={
            "supervised_ensemble": test_supervised_ensemble,
            "isolation_forest": test_scores.get("isolation_forest_unsupervised", np.zeros(len(test_df))),
        },
        ensemble_score=test_final,
        warning_threshold=threshold,
        critical_threshold=critical_threshold,
        alert_mode=main_alert_mode,
        rule_score=df_model["rule_score"] if "rule_score" in df_model else None,
    )
    predictions.to_csv(tables_dir / "anomaly_predictions_test.csv", index=False)

    # When requested, also save separate balanced/conservative prediction files.
    if args.alert_mode == "both":
        for mode_name, mode_threshold in threshold_map.items():
            mode_critical = critical_threshold_from_validation(
                val_final, mode_threshold, percentile=args.critical_percentile
            )
            mode_predictions = create_prediction_table(
                test_df,
                args.target,
                scores={
                    "supervised_ensemble": test_supervised_ensemble,
                    "isolation_forest": test_scores.get("isolation_forest_unsupervised", np.zeros(len(test_df))),
                },
                ensemble_score=test_final,
                warning_threshold=mode_threshold,
                critical_threshold=mode_critical,
                alert_mode=mode_name,
                rule_score=df_model["rule_score"] if "rule_score" in df_model else None,
            )
            mode_predictions.to_csv(tables_dir / f"anomaly_predictions_test_{mode_name}.csv", index=False)

    # Explain top high-risk rows.
    print("Generating z-score explanations...")
    explanations = top_zscore_explanations(
        test_df,
        numeric_cols,
        normal_train_df,
        predictions,
        max_rows=args.explain_top_n,
    )
    for col in ["top_1_feature", "top_2_feature", "top_3_feature"]:
        if col in explanations.columns:
            explanations[col + "_label"] = explanations[col].map(readable_feature_name)
    
    explanations.to_csv(tables_dir / "anomaly_reason_ranking.csv", index=False)

    lead = early_warning_summary(predictions, test_df, args.target)
    if not lead.empty:
        lead.to_csv(tables_dir / "early_warning_summary_by_session.csv", index=False)
        lead_overall = pd.DataFrame([{
            "sessions_with_anomaly": len(lead),
            "sessions_alerted": int(lead["first_alert_time_s"].notna().sum()),
            "alerted_before_or_at_start_rate": float(lead["alerted_before_or_at_start"].mean()),
            "median_lead_time_s": float(lead["lead_time_s"].median(skipna=True)),
            "mean_lead_time_s": float(lead["lead_time_s"].mean(skipna=True)),
        }])
        lead_overall.to_csv(tables_dir / "early_warning_overall.csv", index=False)

    # Save feature importance from random forest if available.
    for res in model_results:
        if res["name"] == "random_forest":
            save_feature_importance(res["pipeline"], tables_dir)
            break

    if not args.no_plots:
        print("Generating result plots...")
        generate_result_plots(
            figures_dir=figures_dir,
            tables_dir=tables_dir,
            comparison=comparison,
            predictions=predictions,
            explanations=explanations,
            target=args.target,
        )

    # Save a concise text report.
    best_test = comparison[comparison["dataset"].eq("test")].sort_values("f1", ascending=False).head(1)
    report_path = output_dir / "run_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Kempower Anomaly Ensemble Run Summary\n\n")
        f.write(f"Data: `{data_path}`\n\n")
        f.write(f"Target: `{args.target}`\n\n")
        f.write(f"Alert mode: `{main_alert_mode}`\n\n")
        f.write(f"Warning threshold: `{threshold:.6f}`\n\n")
        f.write(f"Critical threshold: `{critical_threshold:.6f}`\n\n")
        f.write(f"Rows loaded: {len(df):,}\n\n")
        f.write("## Split Summary\n\n")
        f.write(dataframe_to_markdown_safe(split_summary, index=False))
        f.write("\n\n## Best Test Model by F1\n\n")
        f.write(dataframe_to_markdown_safe(best_test, index=False))
        f.write("\n\n## Threshold Mode Comparison\n\n")
        f.write(dataframe_to_markdown_safe(threshold_modes[threshold_modes["dataset"].eq("test")], index=False))
        f.write("\n\n## Output Files\n\n")
        f.write("- `tables/overall_anomaly_rates.csv`\n")
        f.write("- `tables/rates_by_scenario.csv`\n")
        f.write("- `tables/model_comparison.csv`\n")
        f.write("- `tables/anomaly_predictions_test.csv`\n")
        f.write("- `tables/anomaly_reason_ranking.csv`\n")
        f.write("- `tables/threshold_modes.csv`\n")
        f.write("- `figures/*.png`\n")
        f.write("- `models/*.joblib`\n")

    print(f"\nDone. Results written to: {output_dir}")
    print(f"Main predictions: {tables_dir / 'anomaly_predictions_test.csv'}")
    print(f"Model comparison: {tables_dir / 'model_comparison.csv'}")
    print(f"Rate reports: {tables_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kempower telemetry anomaly-rate assessment and ensemble prediction pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to telemetry CSV")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target label column")
    parser.add_argument("--output-dir", type=str, default="results_kempower_anomaly", help="Output directory")
    parser.add_argument("--window-samples", type=int, default=6, help="Rolling window samples; 6 = 60s at 10s sampling")
    parser.add_argument("--sample-seconds", type=int, default=10, help="Sampling interval in seconds")
    parser.add_argument("--include-asset-id-features", action="store_true", help="Allow stationId/chargerId as ML categorical features")
    parser.add_argument("--fast", action="store_true", help="Use smaller/faster models")
    parser.add_argument("--explain-top-n", type=int, default=1000, help="Number of highest-risk rows to explain")
    parser.add_argument(
        "--alert-mode",
        choices=["balanced", "conservative", "both"],
        default="balanced",
        help="Balanced maximizes validation F1; conservative limits validation false-alarm rate; both saves both prediction tables.",
    )
    parser.add_argument(
        "--conservative-fpr",
        type=float,
        default=0.01,
        help="Maximum validation false-alarm rate for conservative mode.",
    )
    parser.add_argument(
        "--critical-percentile",
        type=float,
        default=95.0,
        help="Percentile of high-risk validation scores used to set the Critical threshold.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable automatic PNG figure generation")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
