"""Integration tests for shared machine learning contracts."""

from datetime import UTC, datetime, timedelta

import pytest

from taqamind.core.interfaces import (
    AcceptancePolicy,
    EvaluationDecision,
    FinalEvaluationResult,
    FrozenModelSelection,
    LeakageCheck,
    StageTiming,
    TrainingAuthorization,
    TrainingRunResult,
    validate_training_request,
)
from taqamind.core.interfaces import (
    TestMetricResult as ProtectedTestMetrics,
)
from tests.core.interfaces.test_candidates import (
    make_successful_result,
)
from tests.core.interfaces.test_evaluation import make_metrics
from tests.core.interfaces.test_leakage import make_validation
from tests.core.interfaces.test_training import make_configuration


def test_ml_contract_workflow():
    leakage = make_validation()

    authorization = TrainingAuthorization.from_validation(
        leakage
    )

    development = authorization.development_split

    configuration = make_configuration()

    policy = AcceptancePolicy(
        minimum_improvement_pct=10,
        justification="Synthetic integration policy.",
        created_at=(
            development.context.created_at
            - timedelta(seconds=1)
        ),
    )

    validate_training_request(
        authorization=authorization,
        contract=leakage.readiness.contract,
        configuration=configuration,
        policy=policy,
    )

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

    manifest = leakage.manifest

    baseline_test = ProtectedTestMetrics(
        candidate_id=baseline.candidate_id,
        model_artifact_id=(
            baseline.model_reference.artifact_id
        ),
        record_ids=manifest.test.record_ids,
        metrics=make_metrics(12.0),
    )

    challenger_test = ProtectedTestMetrics(
        candidate_id=challenger.candidate_id,
        model_artifact_id=(
            challenger.model_reference.artifact_id
        ),
        record_ids=manifest.test.record_ids,
        metrics=make_metrics(8.0),
    )

    evaluation = FinalEvaluationResult(
        selection=selection,
        split_manifest=manifest,
        baseline_result=baseline_test,
        selected_result=challenger_test,
        timing=StageTiming(
            started_at=selection.selected_at,
            finished_at=datetime.now(UTC),
        ),
    )

    assert leakage.eligible_for_training

    assert run.eligible_for_final_evaluation

    assert evaluation.decision == EvaluationDecision.ACCEPTED

    assert evaluation.eligible_for_registry_validation

    assert evaluation.test_improvement_pct == pytest.approx(
        33.3333333333
    )


def test_leakage_stops_training_workflow():
    leakage = make_validation(
        failed_check=LeakageCheck.HISTORICAL_USAGE,
    )

    assert not leakage.eligible_for_training

    with pytest.raises(ValueError):
        TrainingAuthorization.from_validation(leakage)


def test_unverified_records_stop_training_workflow():
    leakage = make_validation(
        unverified_check=LeakageCheck.RECORD_IDENTITY,
    )

    assert not leakage.eligible_for_training

    with pytest.raises(ValueError):
        TrainingAuthorization.from_validation(leakage)