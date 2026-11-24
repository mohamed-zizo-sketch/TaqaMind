"""Tests for model candidate results and validation selection."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    CandidateModelReference,
    CandidateResult,
    CandidateStatus,
    DevelopmentSplit,
    ExecutionError,
    RegressionMetric,
    StageTiming,
    TrainingRunResult,
    ValidationMetric,
)
from tests.core.interfaces.test_splitting import make_manifest
from tests.core.interfaces.test_training import make_configuration


def make_timing(context):
    return StageTiming(
        started_at=context.created_at,
        finished_at=datetime.now(UTC),
    )


def make_successful_result(
    config,
    development,
    mae,
    **overrides,
):
    context = development.context

    data = {
        "candidate_id": config.candidate_id,
        "status": CandidateStatus.SUCCEEDED,
        "timing": make_timing(context),
        "validation_metrics": (
            ValidationMetric(
                metric=RegressionMetric.MAE,
                value=mae,
            ),
            ValidationMetric(
                metric=RegressionMetric.RMSE,
                value=mae + 1,
            ),
            ValidationMetric(
                metric=RegressionMetric.WAPE,
                value=20.0,
            ),
            ValidationMetric(
                metric=RegressionMetric.R2,
                value=0.7,
            ),
        ),
        "evaluated_row_count": development.validation.row_count,
        "model_reference": CandidateModelReference(
            factory_id=context.factory_id,
            execution_id=context.execution_id,
            candidate_id=config.candidate_id,
            content_sha256="a" * 64,
            size_bytes=1024,
        ),
    }

    data.update(overrides)

    return CandidateResult(**data)


def make_run(
    baseline_mae=10.0,
    challenger_mae=7.0,
):
    config = make_configuration()

    development = DevelopmentSplit.from_manifest(
        make_manifest()
    )

    return TrainingRunResult(
        configuration=config,
        development_split=development,
        results=(
            make_successful_result(
                config.baselines[0],
                development,
                baseline_mae,
            ),
            make_successful_result(
                config.challengers[0],
                development,
                challenger_mae,
            ),
        ),
    )


def test_valid_mae():
    metric = ValidationMetric(
        metric=RegressionMetric.MAE,
        value=5.5,
    )

    assert metric.value == 5.5


@pytest.mark.parametrize(
    "metric",
    [
        RegressionMetric.MAE,
        RegressionMetric.RMSE,
        RegressionMetric.WAPE,
    ],
)
def test_negative_error_metric_rejected(metric):
    with pytest.raises(ValidationError):
        ValidationMetric(
            metric=metric,
            value=-1,
        )


def test_negative_r2_is_valid():
    metric = ValidationMetric(
        metric=RegressionMetric.R2,
        value=-0.5,
    )

    assert metric.value == -0.5


def test_r2_above_one_rejected():
    with pytest.raises(ValidationError):
        ValidationMetric(
            metric=RegressionMetric.R2,
            value=1.5,
        )


def test_undefined_metric_requires_reason():
    with pytest.raises(ValidationError):
        ValidationMetric(
            metric=RegressionMetric.WAPE,
            value=None,
        )


def test_undefined_metric_is_representable():
    metric = ValidationMetric(
        metric=RegressionMetric.WAPE,
        value=None,
        undefined_reason="Zero target denominator.",
    )

    assert metric.value is None


def test_nonfinite_metric_rejected():
    with pytest.raises(ValidationError):
        ValidationMetric(
            metric=RegressionMetric.MAE,
            value=float("inf"),
        )


def test_successful_training_run():
    run = make_run()

    assert len(run.successful_results) == 2

    assert run.best_baseline.get_metric(
        RegressionMetric.MAE
    ) == 10.0

    assert run.best_challenger.get_metric(
        RegressionMetric.MAE
    ) == 7.0

    assert run.eligible_for_final_evaluation


def test_baseline_improvement():
    run = make_run()

    assert run.validation_improvement_pct == pytest.approx(30.0)


def test_challenger_worse_than_baseline():
    run = make_run(
        baseline_mae=5.0,
        challenger_mae=8.0,
    )

    assert not run.eligible_for_final_evaluation

    assert run.validation_improvement_pct < 0


def test_equal_performance_is_not_improvement():
    run = make_run(
        baseline_mae=5.0,
        challenger_mae=5.0,
    )

    assert not run.eligible_for_final_evaluation


def test_zero_baseline_improvement_is_undefined():
    run = make_run(
        baseline_mae=0.0,
        challenger_mae=1.0,
    )

    assert run.validation_improvement_pct is None

    assert not run.eligible_for_final_evaluation


def test_success_requires_model_reference():
    run = make_run()

    payload = run.results[0].model_dump()
    payload["model_reference"] = None

    with pytest.raises(ValidationError):
        CandidateResult.model_validate(payload)


def test_success_requires_mae():
    run = make_run()

    payload = run.results[0].model_dump()

    payload["validation_metrics"] = tuple(
        metric
        for metric in payload["validation_metrics"]
        if metric["metric"] != RegressionMetric.MAE
    )

    with pytest.raises(ValidationError):
        CandidateResult.model_validate(payload)


def test_failed_candidate_requires_error():
    run = make_run()

    with pytest.raises(ValidationError):
        CandidateResult(
            candidate_id=run.results[0].candidate_id,
            status=CandidateStatus.FAILED,
            timing=run.results[0].timing,
        )


def test_failed_baseline_blocks_final_evaluation():
    run = make_run()

    baseline = run.results[0]

    failed = CandidateResult(
        candidate_id=baseline.candidate_id,
        status=CandidateStatus.FAILED,
        timing=baseline.timing,
        error=ExecutionError(
            code="TRAINING_FAILED",
            message="Synthetic training failure.",
        ),
    )

    updated = TrainingRunResult(
        configuration=run.configuration,
        development_split=run.development_split,
        results=(
            failed,
            run.results[1],
        ),
    )

    assert not updated.eligible_for_final_evaluation


def test_missing_candidate_result_rejected():
    run = make_run()

    with pytest.raises(
        ValidationError,
        match="account for every candidate",
    ):
        TrainingRunResult(
            configuration=run.configuration,
            development_split=run.development_split,
            results=(run.results[0],),
        )


def test_duplicate_candidate_result_rejected():
    run = make_run()

    with pytest.raises(ValidationError):
        TrainingRunResult(
            configuration=run.configuration,
            development_split=run.development_split,
            results=(
                run.results[0],
                run.results[0],
            ),
        )


def test_candidate_factory_mismatch():
    run = make_run()

    payload = run.model_dump()

    payload["results"][0]["model_reference"]["factory_id"] = uuid4()

    with pytest.raises(
        ValidationError,
        match="factory mismatch",
    ):
        TrainingRunResult.model_validate(payload)


def test_candidate_execution_mismatch():
    run = make_run()

    payload = run.model_dump()

    payload["results"][0]["model_reference"]["execution_id"] = uuid4()

    with pytest.raises(
        ValidationError,
        match="execution mismatch",
    ):
        TrainingRunResult.model_validate(payload)


def test_validation_row_count_mismatch():
    run = make_run()

    payload = run.model_dump()

    payload["results"][0]["evaluated_row_count"] = 999

    with pytest.raises(
        ValidationError,
        match="row count mismatch",
    ):
        TrainingRunResult.model_validate(payload)


def test_skipped_candidate_requires_reason():
    run = make_run()

    with pytest.raises(ValidationError):
        CandidateResult(
            candidate_id=run.results[0].candidate_id,
            status=CandidateStatus.SKIPPED,
            timing=run.results[0].timing,
        )


def test_training_run_serialization():
    run = make_run()

    restored = TrainingRunResult.model_validate_json(
        run.model_dump_json()
    )

    assert restored == run