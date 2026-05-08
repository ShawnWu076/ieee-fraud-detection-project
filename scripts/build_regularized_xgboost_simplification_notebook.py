from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "09_regularized_xgboost_simplification.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # 09 Regularized XGBoost Simplification

            Notebook 08 showed a visible train-validation gap, especially for the deeper tuned models.

            This notebook keeps the same full-data time split and the same `v2_full` feature set, then tests whether a slightly simpler XGBoost improves validation behavior.
            """
        ),
        markdown_cell(
            """
            ## Setup

            The comparison is intentionally narrow:

            - `v2_full_reference`: the original full-data reference model
            - `tuned_params_no_class_weight`: the strongest model from notebook 08
            - `tuned_cost_sensitive_xgb`: the cost-sensitive version from notebook 07/08
            - `regularized_depth6_no_weight`: a smaller depth-6 version with stronger regularization
            - `regularized_depth5_no_weight`: a smaller depth-5 version with stronger regularization

            The business threshold section uses `FN:FP = 10:1` in this notebook.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import json
            import sys
            import warnings

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from IPython.display import display
            from sklearn.metrics import confusion_matrix
            from xgboost import XGBClassifier

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            from src.fraud_detection.data_prep_safe import load_merged_data_safe, make_time_validation_split
            from src.fraud_detection.eda import set_plot_theme
            from src.fraud_detection.metrics import compute_classification_metrics
            from src.fraud_detection.tree_preprocessing_v2 import fit_tree_preprocessor_v2, transform_tree_preprocessor_v2

            warnings.filterwarnings("ignore")
            set_plot_theme()
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.float_format", "{:,.4f}".format)

            RANDOM_STATE = 42
            FN_COST = 10
            FP_COST = 1
            THRESHOLDS = np.unique(
                np.r_[
                    np.linspace(0.001, 0.050, 50),
                    np.linspace(0.055, 0.500, 90),
                    np.linspace(0.510, 0.990, 49),
                ]
            )
            MODEL_COLORS = {
                "v2_full_reference": "#2563eb",
                "tuned_params_no_class_weight": "#64748b",
                "tuned_cost_sensitive_xgb": "#dc2626",
                "regularized_depth6_no_weight": "#16a34a",
                "regularized_depth5_no_weight": "#f59e0b",
            }
            OUTPUT_PATH = PROJECT_ROOT / "outputs" / "regularized_xgboost_simplification_summary.json"
            """
        ),
        markdown_cell("## Full-Data Time Split And Features"),
        code_cell(
            """
            train_df, _ = load_merged_data_safe(nrows=None)
            train_part, valid_part = make_time_validation_split(train_df)

            artifacts = fit_tree_preprocessor_v2(
                train_part,
                add_missing_indicators=True,
                add_group_amount_features=True,
                drop_missing_threshold=0.999,
            )

            x_train = transform_tree_preprocessor_v2(train_part, artifacts, impute_numeric=False)
            x_valid = transform_tree_preprocessor_v2(valid_part, artifacts, impute_numeric=False)
            y_train = train_part["isFraud"].to_numpy()
            y_valid = valid_part["isFraud"].to_numpy()

            positive_count = int(y_train.sum())
            negative_count = int(len(y_train) - positive_count)
            base_scale_pos_weight = negative_count / max(1, positive_count)

            setup_summary = pd.DataFrame(
                [
                    ["labelled_rows", len(train_df)],
                    ["train_rows", len(train_part)],
                    ["validation_rows", len(valid_part)],
                    ["train_fraud_rate", y_train.mean()],
                    ["validation_fraud_rate", y_valid.mean()],
                    ["feature_count", x_train.shape[1]],
                    ["base_scale_pos_weight", base_scale_pos_weight],
                ],
                columns=["item", "value"],
            )
            display(setup_summary.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Candidate Models"),
        code_cell(
            """
            def make_xgb(params):
                return XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    **params,
                )


            model_configs = {
                "v2_full_reference": {
                    "n_estimators": 400,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                },
                "tuned_params_no_class_weight": {
                    "n_estimators": 700,
                    "max_depth": 8,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 3,
                    "reg_lambda": 2.0,
                    "reg_alpha": 0.1,
                    "gamma": 0.0,
                    "scale_pos_weight": 1.0,
                },
                "tuned_cost_sensitive_xgb": {
                    "n_estimators": 700,
                    "max_depth": 8,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.7,
                    "min_child_weight": 3,
                    "reg_lambda": 2.0,
                    "reg_alpha": 0.1,
                    "gamma": 0.0,
                    "scale_pos_weight": base_scale_pos_weight * 0.65,
                },
                "regularized_depth6_no_weight": {
                    "n_estimators": 650,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "subsample": 0.82,
                    "colsample_bytree": 0.75,
                    "min_child_weight": 6,
                    "reg_lambda": 6.0,
                    "reg_alpha": 0.25,
                    "gamma": 0.10,
                    "scale_pos_weight": 1.0,
                },
                "regularized_depth5_no_weight": {
                    "n_estimators": 550,
                    "max_depth": 5,
                    "learning_rate": 0.04,
                    "subsample": 0.85,
                    "colsample_bytree": 0.75,
                    "min_child_weight": 8,
                    "reg_lambda": 10.0,
                    "reg_alpha": 0.50,
                    "gamma": 0.25,
                    "scale_pos_weight": 1.0,
                },
            }

            model_summary = pd.DataFrame(
                [
                    {
                        "model": model_name,
                        "n_estimators": params.get("n_estimators"),
                        "max_depth": params.get("max_depth"),
                        "learning_rate": params.get("learning_rate"),
                        "min_child_weight": params.get("min_child_weight", np.nan),
                        "reg_lambda": params.get("reg_lambda", np.nan),
                        "reg_alpha": params.get("reg_alpha", 0.0),
                        "gamma": params.get("gamma", 0.0),
                        "scale_pos_weight": params.get("scale_pos_weight", 1.0),
                    }
                    for model_name, params in model_configs.items()
                ]
            )
            display(model_summary.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Train And Evaluate"),
        code_cell(
            """
            def threshold_policy_table(y_true, y_score, model_name, thresholds=THRESHOLDS, fn_cost=FN_COST, fp_cost=FP_COST):
                y_true = np.asarray(y_true, dtype=int)
                y_score = np.asarray(y_score, dtype=float)
                rows = []

                for threshold in thresholds:
                    y_pred = (y_score >= threshold).astype(int)
                    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
                    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
                    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
                    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
                    total_cost = fn_cost * fn + fp_cost * fp
                    rows.append(
                        {
                            "model": model_name,
                            "threshold": float(threshold),
                            "true_positives": tp,
                            "false_positives": fp,
                            "false_negatives": fn,
                            "true_negatives": tn,
                            "precision": tp / max(1, tp + fp),
                            "recall": tp / max(1, tp + fn),
                            "review_rate": (tp + fp) / len(y_true),
                            "total_cost": total_cost,
                            "cost_per_1k_txn": total_cost / len(y_true) * 1000,
                        }
                    )

                return pd.DataFrame(rows)


            fitted_models = {}
            score_by_model = {}
            threshold_tables = {}
            metric_rows = []
            generalization_rows = []

            for model_name, params in model_configs.items():
                model = make_xgb(params)
                model.fit(x_train, y_train)

                train_scores = model.predict_proba(x_train)[:, 1]
                valid_scores = model.predict_proba(x_valid)[:, 1]
                fitted_models[model_name] = model
                score_by_model[model_name] = valid_scores

                train_metrics = compute_classification_metrics(
                    y_train,
                    train_scores,
                    top_fractions=(0.01, 0.03, 0.05),
                )
                valid_metrics = compute_classification_metrics(
                    y_valid,
                    valid_scores,
                    top_fractions=(0.01, 0.03, 0.05),
                )

                for split_name, metrics in [("train", train_metrics), ("validation", valid_metrics)]:
                    generalization_rows.append(
                        {
                            "model": model_name,
                            "split": split_name,
                            "roc_auc": metrics["roc_auc"],
                            "pr_auc": metrics["average_precision"],
                            "precision_at_top_3pct": metrics["precision_at_top_3pct"],
                            "recall_at_top_3pct": metrics["recall_at_top_3pct"],
                            "precision_at_top_5pct": metrics["precision_at_top_5pct"],
                            "recall_at_top_5pct": metrics["recall_at_top_5pct"],
                        }
                    )

                threshold_table = threshold_policy_table(y_valid, valid_scores, model_name)
                threshold_tables[model_name] = threshold_table
                best_policy = threshold_table.sort_values(
                    ["total_cost", "false_negatives", "review_rate"],
                    ascending=[True, True, True],
                ).iloc[0]

                metric_rows.append(
                    {
                        "model": model_name,
                        "validation_roc_auc": valid_metrics["roc_auc"],
                        "validation_pr_auc": valid_metrics["average_precision"],
                        "validation_precision_at_top_3pct": valid_metrics["precision_at_top_3pct"],
                        "validation_recall_at_top_3pct": valid_metrics["recall_at_top_3pct"],
                        "validation_precision_at_top_5pct": valid_metrics["precision_at_top_5pct"],
                        "validation_recall_at_top_5pct": valid_metrics["recall_at_top_5pct"],
                        "train_pr_auc": train_metrics["average_precision"],
                        "pr_auc_gap": train_metrics["average_precision"] - valid_metrics["average_precision"],
                        "optimized_threshold": best_policy["threshold"],
                        "optimized_false_positives": best_policy["false_positives"],
                        "optimized_false_negatives": best_policy["false_negatives"],
                        "optimized_recall": best_policy["recall"],
                        "optimized_review_rate": best_policy["review_rate"],
                        "optimized_cost_per_1k_txn": best_policy["cost_per_1k_txn"],
                    }
                )

            model_results = pd.DataFrame(metric_rows).sort_values(
                ["validation_pr_auc", "optimized_cost_per_1k_txn"],
                ascending=[False, True],
            )
            generalization_metrics = pd.DataFrame(generalization_rows)

            display(model_results.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Train vs Validation Gap"),
        code_cell(
            """
            display(
                generalization_metrics.sort_values(["model", "split"]).style.hide(axis="index")
            )

            gap_plot = generalization_metrics.melt(
                id_vars=["model", "split"],
                value_vars=["roc_auc", "pr_auc", "precision_at_top_3pct", "recall_at_top_3pct"],
                var_name="metric",
                value_name="value",
            )

            g = sns.catplot(
                data=gap_plot,
                x="split",
                y="value",
                hue="model",
                col="metric",
                kind="bar",
                palette=MODEL_COLORS,
                col_wrap=2,
                height=3.5,
                aspect=1.25,
                sharey=False,
                legend=False,
            )
            g.set_titles("{col_name}")
            g.set_axis_labels("", "score")
            handles, labels = g.axes.flat[0].get_legend_handles_labels()
            g.fig.legend(
                handles,
                labels,
                title="model",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.03),
                ncol=3,
                frameon=False,
            )
            g.fig.subplots_adjust(top=0.86, hspace=0.35, wspace=0.20)
            plt.show()
            """
        ),
        markdown_cell("## Validation Ranking And Cost"),
        code_cell(
            """
            plot_metrics = model_results.melt(
                id_vars=["model"],
                value_vars=[
                    "validation_pr_auc",
                    "validation_recall_at_top_3pct",
                    "validation_precision_at_top_3pct",
                    "optimized_cost_per_1k_txn",
                ],
                var_name="metric",
                value_name="value",
            )

            g = sns.catplot(
                data=plot_metrics,
                x="value",
                y="model",
                col="metric",
                kind="bar",
                palette=[MODEL_COLORS.get(model, "#64748b") for model in model_results["model"]],
                col_wrap=2,
                height=3.6,
                aspect=1.35,
                sharex=False,
            )
            g.set_titles("{col_name}")
            g.set_axis_labels("", "")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Threshold Curves For The Leading Models"),
        code_cell(
            """
            leading_models = model_results.head(3)["model"].tolist()
            threshold_plot = pd.concat(
                [threshold_tables[model] for model in leading_models],
                ignore_index=True,
            )

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for metric, ax, title in [
                ("recall", axes[0], "Recall by threshold"),
                ("precision", axes[1], "Precision by threshold"),
                ("cost_per_1k_txn", axes[2], f"Cost per 1k txns (FN:{FN_COST}, FP:{FP_COST})"),
            ]:
                sns.lineplot(
                    data=threshold_plot,
                    x="threshold",
                    y=metric,
                    hue="model",
                    palette=MODEL_COLORS,
                    ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("threshold")
                ax.legend(title="")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Optimized Confusion Matrices"),
        code_cell(
            """
            def confusion_counts_at_threshold(y_true, y_score, threshold):
                y_pred = (np.asarray(y_score) >= threshold).astype(int)
                return confusion_matrix(y_true, y_pred, labels=[1, 0])


            leading = model_results.head(3).copy()
            fig, axes = plt.subplots(1, len(leading), figsize=(5.5 * len(leading), 5))

            for ax, row in zip(axes, leading.itertuples(index=False)):
                matrix = confusion_counts_at_threshold(
                    y_valid,
                    score_by_model[row.model],
                    row.optimized_threshold,
                )
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt=",d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["pred_fraud", "pred_legit"],
                    yticklabels=["true_fraud", "true_legit"],
                    ax=ax,
                )
                ax.set_title(f"{row.model}\\nthreshold={row.optimized_threshold:.3f}; top-left=TP")
                ax.set_xlabel("")
                ax.set_ylabel("")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Reading The Result"),
        code_cell(
            """
            best_model = model_results.iloc[0]
            reference = model_results.loc[model_results["model"] == "v2_full_reference"].iloc[0]

            report_summary = pd.DataFrame(
                [
                    {
                        "question": "Best validation PR-AUC model",
                        "answer": best_model["model"],
                    },
                    {
                        "question": "PR-AUC lift vs v2_full_reference",
                        "answer": best_model["validation_pr_auc"] - reference["validation_pr_auc"],
                    },
                    {
                        "question": "Cost per 1k improvement vs v2_full_reference",
                        "answer": reference["optimized_cost_per_1k_txn"] - best_model["optimized_cost_per_1k_txn"],
                    },
                    {
                        "question": "False negatives reduced vs v2_full_reference",
                        "answer": reference["optimized_false_negatives"] - best_model["optimized_false_negatives"],
                    },
                    {
                        "question": "PR-AUC train-validation gap",
                        "answer": best_model["pr_auc_gap"],
                    },
                ]
            )
            display(report_summary.style.hide(axis="index"))

            recommendation = (
                f"Use {best_model['model']} for the current validation-backed policy. "
                "The simplification pass did not replace it unless another candidate ranks first above."
            )
            print(recommendation)

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(
                json.dumps(
                    {
                        "model_results": model_results.to_dict(orient="records"),
                        "generalization_metrics": generalization_metrics.to_dict(orient="records"),
                        "recommendation": recommendation,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
            """
        ),
    ]

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(build_notebook()), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
