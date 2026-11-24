"""Tests for protected final model evaluation."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    AcceptancePolicy,
    DevelopmentSplit,
    EvaluationDecision,
    FinalEvaluationResult,
    FrozenModelSelection,
    RegressionMetric,
    StageTiming,
    TrainingRunResult,
    ValidationMetric,
)
from taqamind.core.interfaces import (
    TestMetricResult as ProtectedTestMetrics,
)
from tests.core.interfaces.test_candidates import (
    make_successful_result,
)
from tests.core.interfaces.test_splitting import make_manifest
from tests.core.interfaces.test_training import make_configuration


def make_metrics(mae):
    return (
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
            value=0.5,
        ),
    )


def make_evaluation(
    baseline_mae=12.0,
    selected_mae=8.0,
    minimum_improvement=10.0,
    maximum_mae=None,
):
    policy = AcceptancePolicy(
        minimum_improvement_pct=minimum_improvement,
        maximum_test_mae=maximum_mae,
        justification="Synthetic acceptance policy.",
        created_at=datetime.now(UTC) - timedelta(seconds=10),
    )

    manifest = make_manifest()

    development = DevelopmentSplit.from_manifest(
        manifest
    )

    configuration = make_configuration()

    baseline = make_successful_result(
        configuration.baselines[0],
        development,
        10.0,
    )

    challenger = make_successful_result(
        configuration.challengers[0],
        development,
        7.0,
    )

    run = TrainingRunResult(
        configuration=configuration,
        development_split=development,
        results=(baseline, challenger),
    )

    selection = FrozenModelSelection.from_training_run(
        training_run=run,
        policy=policy,
    )

    baseline_result = ProtectedTestMetrics(
        candidate_id=baseline.candidate_id,
        model_artifact_id=baseline.model_reference.artifact_id,
        record_ids=manifest.test.record_ids,
        metrics=make_metrics(baseline_mae),
    )

    selected_result = ProtectedTestMetrics(
        candidate_id=challenger.candidate_id,
        model_artifact_id=challenger.model_reference.artifact_id,
        record_ids=manifest.test.record_ids,
        metrics=make_metrics(selected_mae),
    )

    return FinalEvaluationResult(
        selection=selection,
        split_manifest=manifest,
        baseline_result=baseline_result,
        selected_result=selected_result,
        timing=StageTiming(
            started_at=selection.selected_at,
            finished_at=datetime.now(UTC),
        ),
    )


def test_model_is_accepted():
    result = make_evaluation()

    assert result.decision == EvaluationDecision.ACCEPTED
    assert result.eligible_for_registry_validation


def test_improvement_calculation():
    result = make_evaluation()

    assert result.test_improvement_pct == pytest.approx(
        33.3333333333
    )


def test_model_worse_than_baseline_is_rejected():
    result = make_evaluation(
        baseline_mae=5.0,
        selected_mae=8.0,
    )

    assert result.decision == EvaluationDecision.REJECTED

    assert (
        "INSUFFICIENT_BASELINE_IMPROVEMENT"
        in result.rejection_reasons
    )


def test_insufficient_improvement():
    result = make_evaluation(
        baseline_mae=10.0,
        selected_mae=9.8,
        minimum_improvement=10.0,
    )

    assert not result.eligible_for_registry_validation


def test_maximum_mae_requirement():
    result = make_evaluation(
        baseline_mae=12.0,
        selected_mae=8.0,
        maximum_mae=5.0,
    )

    assert result.decision == EvaluationDecision.REJECTED

    assert (
        "MAXIMUM_TEST_MAE_EXCEEDED"
        in result.rejection_reasons
    )


def test_zero_baseline_mae():
    result = make_evaluation(
        baseline_mae=0.0,
        selected_mae=1.0,
    )

    assert result.test_improvement_pct is None
    assert result.decision == EvaluationDecision.REJECTED


def test_wrong_test_records_rejected():
    result = make_evaluation()

    payload = result.model_dump()

    payload["selected_result"]["record_ids"] = (
        uuid4(),
        *result.selected_result.record_ids[1:],
    )

    with pytest.raises(
        ValidationError,
        match="exact protected test records",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_training_records_cannot_be_used_as_test():
    result = make_evaluation()

    payload = result.model_dump()

    payload["selected_result"]["record_ids"] = (
        result.split_manifest.train.record_ids[0],
        *result.selected_result.record_ids[1:],
    )

    with pytest.raises(
        ValidationError,
        match="exact protected test records",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_wrong_candidate_rejected():
    result = make_evaluation()

    payload = result.model_dump()

    payload["selected_result"]["candidate_id"] = uuid4()

    with pytest.raises(
        ValidationError,
        match="frozen selection",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_wrong_model_artifact_rejected():
    result = make_evaluation()

    payload = result.model_dump()

    payload["selected_result"]["model_artifact_id"] = uuid4()

    with pytest.raises(
        ValidationError,
        match="selected model artifact",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_wrong_split_rejected():
    result = make_evaluation()

    payload = result.model_dump()

    payload["split_manifest"]["split_id"] = uuid4()

    with pytest.raises(
        ValidationError,
        match="does not match training split",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_policy_cannot_be_defined_after_training_start():
    result = make_evaluation()

    payload = result.selection.model_dump()

    payload["policy"]["created_at"] = (
        result.selection.selected_at
        + timedelta(seconds=1)
    )

    with pytest.raises(
        ValidationError,
        match="before training",
    ):
        FrozenModelSelection.model_validate(payload)


def test_evaluation_cannot_precede_selection():
    result = make_evaluation()

    payload = result.model_dump()

    payload["timing"]["started_at"] = (
        result.selection.selected_at
        - timedelta(seconds=1)
    )

    with pytest.raises(
        ValidationError,
        match="precede model selection",
    ):
        FinalEvaluationResult.model_validate(payload)


def test_evaluation_serialization():
    result = make_evaluation()

    restored = FinalEvaluationResult.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result


def test_evaluation_is_frozen():
    result = make_evaluation()

    with pytest.raises(ValidationError):
        result.evaluation_id = uuid4()