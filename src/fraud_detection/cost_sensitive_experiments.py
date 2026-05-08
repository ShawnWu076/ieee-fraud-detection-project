from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .data_prep_safe import load_merged_data_safe, make_time_validation_split
from .metrics import compute_classification_metrics
from .tree_preprocessing_v2 import fit_tree_preprocessor_v2, transform_tree_preprocessor_v2

DEFAULT_TOP_FRACTIONS = (0.01, 0.03, 0.05)
PRIMARY_METRIC = "roc_auc"
SECONDARY_METRICS = ("average_precision", "precision_at_top_3pct", "recall_at_top_3pct")
DIAGNOSTIC_METRICS = ("precision_at_top_1pct", "precision_at_top_5pct")


@dataclass
class CostSensitiveDataBundle:
    variant_label: str
    add_missing_indicators: bool
    add_group_amount_features: bool
    train_part: pd.DataFrame
    valid_part: pd.DataFrame
    x_train: pd.DataFrame
    x_valid: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    input_rows: int
    full_train_rows: int
    sample_rows: int
    train_rows: int
    valid_rows: int
    fraud_rate_train: float
    fraud_rate_valid: float
    base_scale_pos_weight: float
    random_state: int


def prepare_cost_sensitive_data(
    sample_size: int | None = 80_000,
    random_state: int = 42,
    nrows: int | None = None,
    *,
    variant_label: str = "v2_full",
    add_missing_indicators: bool = True,
    add_group_amount_features: bool = True,
) -> CostSensitiveDataBundle:
    train_df, _ = load_merged_data_safe(nrows=nrows)
    full_train_rows = int(len(train_df))
    train_part, valid_part = make_time_validation_split(train_df)
    sampled_train_part = train_part
    if sample_size is not None and sample_size < len(train_part):
        sampled_train_part = _time_spaced_sample(train_part, sample_size=sample_size)

    artifacts = fit_tree_preprocessor_v2(
        sampled_train_part,
        add_missing_indicators=add_missing_indicators,
        add_group_amount_features=add_group_amount_features,
        drop_missing_threshold=0.999,
    )

    x_train = transform_tree_preprocessor_v2(sampled_train_part, artifacts, impute_numeric=False)
    x_valid = transform_tree_preprocessor_v2(valid_part, artifacts, impute_numeric=False)
    y_train = sampled_train_part["isFraud"].to_numpy()
    y_valid = valid_part["isFraud"].to_numpy()

    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, int(len(y_train) - positive_count))
    base_scale_pos_weight = negative_count / positive_count

    return CostSensitiveDataBundle(
        variant_label=variant_label,
        add_missing_indicators=add_missing_indicators,
        add_group_amount_features=add_group_amount_features,
        train_part=sampled_train_part,
        valid_part=valid_part,
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        input_rows=full_train_rows,
        full_train_rows=int(len(train_part)),
        sample_rows=int(len(sampled_train_part)),
        train_rows=int(len(sampled_train_part)),
        valid_rows=int(len(valid_part)),
        fraud_rate_train=float(np.mean(y_train)),
        fraud_rate_valid=float(np.mean(y_valid)),
        base_scale_pos_weight=float(base_scale_pos_weight),
        random_state=random_state,
    )


def fit_cost_sensitive_xgboost(
    data_bundle: CostSensitiveDataBundle,
    *,
    pos_weight_multiplier: float = 1.0,
    param_overrides: dict[str, float | int] | None = None,
    label: str | None = None,
    return_artifacts: bool = False,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
) -> dict[str, object]:
    params = build_xgboost_params(
        random_state=data_bundle.random_state,
        scale_pos_weight=data_bundle.base_scale_pos_weight * pos_weight_multiplier,
    )
    if param_overrides:
        params.update(param_overrides)

    model = XGBClassifier(**params)
    model.fit(data_bundle.x_train, data_bundle.y_train)
    valid_scores = model.predict_proba(data_bundle.x_valid)[:, 1]
    metrics = compute_classification_metrics(
        data_bundle.y_valid,
        valid_scores,
        top_fractions=top_fractions,
    )
    effective_scale_pos_weight = float(params["scale_pos_weight"])
    effective_multiplier = effective_scale_pos_weight / data_bundle.base_scale_pos_weight

    row: dict[str, object] = {
        "label": label or "run",
        "pos_weight_multiplier": float(effective_multiplier),
        "scale_pos_weight": effective_scale_pos_weight,
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": float(params["min_child_weight"]),
        "reg_lambda": float(params["reg_lambda"]),
        "reg_alpha": float(params["reg_alpha"]),
        "gamma": float(params["gamma"]),
        **metrics,
    }

    result: dict[str, object] = {
        "row": row,
        "model": model,
        "valid_scores": valid_scores,
        "params": params,
    }
    if return_artifacts:
        result["x_valid"] = data_bundle.x_valid
        result["y_valid"] = data_bundle.y_valid
    return result


def run_weight_sweep(
    data_bundle: CostSensitiveDataBundle,
    multipliers: list[float],
    *,
    base_param_overrides: dict[str, float | int] | None = None,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    artifacts_by_label: dict[str, dict[str, object]] = {}

    for multiplier in multipliers:
        label = f"weight_x{multiplier:g}"
        result = fit_cost_sensitive_xgboost(
            data_bundle,
            pos_weight_multiplier=multiplier,
            param_overrides=base_param_overrides,
            label=label,
            return_artifacts=True,
            top_fractions=top_fractions,
        )
        records.append(result["row"])
        artifacts_by_label[label] = result

    results = rank_results(pd.DataFrame(records))
    results = annotate_weight_stability(results, primary_metric=PRIMARY_METRIC)
    return results, artifacts_by_label


def run_fine_tuning_grid(
    data_bundle: CostSensitiveDataBundle,
    *,
    pos_weight_multiplier: float = 1.0,
    param_grid: dict[str, list[float | int]],
    fixed_params: dict[str, float | int] | None = None,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    grid_keys = list(param_grid)
    records: list[dict[str, object]] = []
    artifacts_by_label: dict[str, dict[str, object]] = {}

    for values in product(*(param_grid[key] for key in grid_keys)):
        overrides = dict(zip(grid_keys, values))
        if fixed_params:
            overrides = {**fixed_params, **overrides}

        short_label = ", ".join(f"{key}={value}" for key, value in overrides.items())
        result = fit_cost_sensitive_xgboost(
            data_bundle,
            pos_weight_multiplier=pos_weight_multiplier,
            param_overrides=overrides,
            label=short_label,
            return_artifacts=True,
            top_fractions=top_fractions,
        )
        records.append(result["row"])
        artifacts_by_label[short_label] = result

    results = rank_results(pd.DataFrame(records))
    return results, artifacts_by_label


def run_named_configs(
    data_bundle: CostSensitiveDataBundle,
    *,
    configs: list[dict[str, object]],
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    artifacts_by_label: dict[str, dict[str, object]] = {}

    for config in configs:
        label = str(config["label"])
        pos_weight_multiplier = float(config.get("pos_weight_multiplier", 1.0))
        param_overrides = dict(config.get("params", {}))

        result = fit_cost_sensitive_xgboost(
            data_bundle,
            pos_weight_multiplier=pos_weight_multiplier,
            param_overrides=param_overrides,
            label=label,
            return_artifacts=True,
            top_fractions=top_fractions,
        )
        records.append(result["row"])
        artifacts_by_label[label] = result

    results = rank_results(pd.DataFrame(records))
    return results, artifacts_by_label


def build_run_summary(data_bundle: CostSensitiveDataBundle) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "variant_label", "value": data_bundle.variant_label},
            {"item": "uses_missing_flags", "value": data_bundle.add_missing_indicators},
            {"item": "uses_group_amount_features", "value": data_bundle.add_group_amount_features},
            {"item": "split_strategy", "value": "full_time_split_then_sample_train_only"},
            {"item": "input_rows", "value": data_bundle.input_rows},
            {"item": "full_train_rows", "value": data_bundle.full_train_rows},
            {"item": "sampled_train_rows", "value": data_bundle.sample_rows},
            {"item": "train_rows", "value": data_bundle.train_rows},
            {"item": "validation_rows", "value": data_bundle.valid_rows},
            {"item": "train_fraud_rate", "value": data_bundle.fraud_rate_train},
            {"item": "validation_fraud_rate", "value": data_bundle.fraud_rate_valid},
            {"item": "base_scale_pos_weight", "value": data_bundle.base_scale_pos_weight},
            {"item": "feature_count", "value": data_bundle.x_train.shape[1]},
        ]
    )


def build_shap_diagnostics(
    model: XGBClassifier,
    x_valid: pd.DataFrame,
    y_valid,
    *,
    sample_size: int = 1_200,
    random_state: int = 42,
    top_n: int = 15,
) -> dict[str, object]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("`shap` is required for SHAP diagnostics.") from exc

    if len(x_valid) <= sample_size:
        x_sample = x_valid.copy()
    else:
        x_sample = x_valid.sample(n=sample_size, random_state=random_state).sort_index()

    y_sample = pd.Series(np.asarray(y_valid), index=x_valid.index, name="isFraud").loc[x_sample.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[-1]
    if isinstance(expected_value, np.ndarray):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]

    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[-1])
    elif hasattr(shap_values, "values"):
        shap_matrix = np.asarray(shap_values.values)
    else:
        shap_matrix = np.asarray(shap_values)

    importance = (
        pd.DataFrame(
            {
                "feature": x_sample.columns,
                "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    candidate_features = [
        feature
        for feature in importance["feature"]
        if pd.to_numeric(x_sample[feature], errors="coerce").notna().sum() >= 50
    ]
    top_feature = candidate_features[0] if candidate_features else importance.iloc[0]["feature"]
    feature_idx = x_sample.columns.get_loc(top_feature)

    dependence_frame = (
        pd.DataFrame(
            {
                top_feature: pd.to_numeric(x_sample[top_feature], errors="coerce"),
                "shap_value": shap_matrix[:, feature_idx],
                "isFraud": y_sample.to_numpy(),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    return {
        "x_sample": x_sample,
        "shap_matrix": shap_matrix,
        "importance": importance,
        "dependence_frame": dependence_frame,
        "top_feature": top_feature,
        "row_count": int(len(x_sample)),
        "expected_value": float(expected_value),
    }


def build_xgboost_params(
    *,
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
) -> dict[str, float | int | str]:
    return {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "min_child_weight": 5,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": random_state,
        "n_jobs": -1,
    }


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    priority_columns = [
        PRIMARY_METRIC,
        *SECONDARY_METRICS,
        *DIAGNOSTIC_METRICS,
    ]
    available_columns = [column for column in priority_columns if column in results.columns]
    return results.sort_values(available_columns, ascending=False).reset_index(drop=True)


def annotate_weight_stability(
    results: pd.DataFrame,
    *,
    primary_metric: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    if primary_metric not in results.columns:
        return results

    ordered = results.sort_values("pos_weight_multiplier").reset_index(drop=True).copy()
    local_means: list[float] = []
    local_ranges: list[float] = []

    for idx in range(len(ordered)):
        start = max(0, idx - 1)
        end = min(len(ordered), idx + 2)
        window = ordered.loc[start : end - 1, primary_metric]
        local_means.append(float(window.mean()))
        local_ranges.append(float(window.max() - window.min()))

    ordered["local_primary_mean"] = local_means
    ordered["local_primary_range"] = local_ranges
    return ordered.sort_values(
        [primary_metric, "local_primary_mean", "local_primary_range", *SECONDARY_METRICS, *DIAGNOSTIC_METRICS],
        ascending=[False, False, True, False, False, False, False, False],
    ).reset_index(drop=True)


def build_stage1_grid(
    data_bundle: CostSensitiveDataBundle,
    *,
    multipliers: tuple[float, ...] = (0.3, 0.5, 0.75, 1.0),
) -> dict[str, list[float | int]]:
    return {
        "n_estimators": [300, 450, 600],
        "max_depth": [4, 5, 6],
        "min_child_weight": [3, 5, 8],
        "scale_pos_weight": [data_bundle.base_scale_pos_weight * multiplier for multiplier in multipliers],
    }


def build_stage2_search_space() -> dict[str, list[float | int]]:
    return {
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "n_estimators": [400, 600, 800, 1000],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "gamma": [0.0, 0.5, 1.0],
    }


def narrow_stage2_grid(
    best_stage1_row: pd.Series | dict[str, object],
    *,
    search_space: dict[str, list[float | int]] | None = None,
) -> dict[str, list[float | int]]:
    base = dict(best_stage1_row)
    search_space = search_space or build_stage2_search_space()

    return {
        "learning_rate": search_space["learning_rate"][:3],
        "n_estimators": _nearest_values(
            search_space["n_estimators"],
            int(base.get("n_estimators", 450)),
            keep=3,
        ),
        "subsample": _nearest_values(
            search_space["subsample"],
            float(base.get("subsample", 0.8)),
            keep=2,
        ),
        "colsample_bytree": _nearest_values(
            search_space["colsample_bytree"],
            float(base.get("colsample_bytree", 0.8)),
            keep=2,
        ),
        "reg_lambda": search_space["reg_lambda"][:3],
        "reg_alpha": search_space["reg_alpha"][:3],
        "gamma": search_space["gamma"][:2],
    }


def _nearest_values(values: list[float | int], center: float, *, keep: int) -> list[float | int]:
    ranked = sorted(values, key=lambda value: (abs(float(value) - center), float(value)))
    chosen = sorted(ranked[:keep], key=float)
    return chosen


def _time_spaced_sample(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if "TransactionDT" not in frame.columns:
        return frame.sample(n=sample_size, random_state=42).reset_index(drop=True)

    ordered = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=sample_size, dtype=int)
    sampled = ordered.iloc[np.unique(positions)].copy()
    return sampled.reset_index(drop=True)
