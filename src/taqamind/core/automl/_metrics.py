"""Shared regression metric computation for Core AutoML and Protected Evaluation."""

from __future__ import annotations

import math

import numpy as np

from taqamind.core.interfaces.candidates import ValidationMetric
from taqamind.core.interfaces.training import RegressionMetric


def compute_regression_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: tuple[RegressionMetric, ...],
) -> tuple[ValidationMetric, ...]:
    """Compute requested regression metrics with explicit undefined guards."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.shape != y_p.shape or y_t.size == 0:
        raise ValueError("Target and prediction arrays must be non-empty and match in shape.")
    if not np.all(np.isfinite(y_t)) or not np.all(np.isfinite(y_p)):
        raise ValueError("Target and prediction values must be finite.")

    results: list[ValidationMetric] = []
    errors = y_t - y_p

    for metric in metrics:
        if metric == RegressionMetric.MAE:
            val = float(np.mean(np.abs(errors)))
            results.append(ValidationMetric(metric=metric, value=max(0.0, val)))
        elif metric == RegressionMetric.RMSE:
            val = float(math.sqrt(float(np.mean(errors**2))))
            results.append(ValidationMetric(metric=metric, value=max(0.0, val)))
        elif metric == RegressionMetric.WAPE:
            denom = float(np.sum(np.abs(y_t)))
            if denom == 0.0:
                results.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="ZERO_ACTUAL_SUM",
                    )
                )
            else:
                val = float(np.sum(np.abs(errors)) / denom)
                results.append(ValidationMetric(metric=metric, value=max(0.0, val)))
        elif metric == RegressionMetric.R2:
            if y_t.size < 2:
                results.append(
                    ValidationMetric(
                        metric=metric,
                        value=None,
                        undefined_reason="INSUFFICIENT_SAMPLES_FOR_R2",
                    )
                )
            else:
                ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
                if ss_tot == 0.0:
                    results.append(
                        ValidationMetric(
                            metric=metric,
                            value=None,
                            undefined_reason="CONSTANT_TARGET_VALUES",
                        )
                    )
                else:
                    ss_res = float(np.sum(errors**2))
                    r2_val = min(1.0, 1.0 - (ss_res / ss_tot))
                    results.append(ValidationMetric(metric=metric, value=r2_val))
    return tuple(results)
