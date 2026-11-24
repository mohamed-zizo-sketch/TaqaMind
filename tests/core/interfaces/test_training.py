"""Tests for controlled training configuration contracts."""

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import TaskType
from taqamind.core.interfaces import (
    CandidateConfiguration,
    CandidateRole,
    DevelopmentSplit,
    ModelFamily,
    ModelParameter,
    RegressionMetric,
    TrainingBudget,
    TrainingConfiguration,
)
from tests.core.interfaces.test_readiness import make_contract
from tests.core.interfaces.test_splitting import make_manifest


def make_baseline():
    return CandidateConfiguration(
        family=ModelFamily.DUMMY_REGRESSOR,
        role=CandidateRole.BASELINE,
        parameters=(
            ModelParameter(
                name="strategy",
                value="median",
            ),
        ),
    )


def make_challenger():
    return CandidateConfiguration(
        family=ModelFamily.RIDGE,
        role=CandidateRole.CHALLENGER,
        parameters=(
            ModelParameter(
                name="alpha",
                value=1.0,
            ),
        ),
    )


def make_configuration(**overrides):
    data = {
        "candidates": (
            make_baseline(),
            make_challenger(),
        ),
        "justification": "Synthetic training configuration.",
    }

    data.update(overrides)

    return TrainingConfiguration(**data)


def test_valid_training_configuration():
    config = make_configuration()

    assert len(config.baselines) == 1
    assert len(config.challengers) == 1

    assert config.primary_metric == RegressionMetric.MAE


def test_model_parameter_map():
    candidate = make_challenger()

    assert candidate.parameter_map == {
        "alpha": 1.0,
    }


def test_duplicate_model_parameters_rejected():
    with pytest.raises(ValidationError):
        CandidateConfiguration(
            family=ModelFamily.RIDGE,
            role=CandidateRole.CHALLENGER,
            parameters=(
                ModelParameter(name="alpha", value=1.0),
                ModelParameter(name="alpha", value=2.0),
            ),
        )


def test_invalid_baseline_role():
    with pytest.raises(ValidationError):
        CandidateConfiguration(
            family=ModelFamily.DUMMY_REGRESSOR,
            role=CandidateRole.CHALLENGER,
        )


def test_invalid_challenger_role():
    with pytest.raises(ValidationError):
        CandidateConfiguration(
            family=ModelFamily.RIDGE,
            role=CandidateRole.BASELINE,
        )


def test_dummy_baseline_is_mandatory():
    with pytest.raises(
        ValidationError,
        match="baseline is mandatory",
    ):
        make_configuration(
            candidates=(
                make_challenger(),
                make_challenger(),
            ),
        )


def test_challenger_is_mandatory():
    with pytest.raises(
        ValidationError,
        match="challenger model is required",
    ):
        make_configuration(
            candidates=(
                make_baseline(),
                make_baseline(),
            ),
        )


def test_duplicate_candidate_ids():
    candidate = make_challenger()

    with pytest.raises(
        ValidationError,
        match="identifiers must be unique",
    ):
        make_configuration(
            candidates=(
                make_baseline(),
                candidate,
                candidate,
            ),
        )


def test_trial_budget_limit():
    with pytest.raises(
        ValidationError,
        match="exceeds training trial budget",
    ):
        make_configuration(
            budget=TrainingBudget(max_trials=1),
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_trials",
        "max_training_seconds",
        "max_parallel_jobs",
        "max_memory_mb",
    ],
)
def test_invalid_budget(field):
    with pytest.raises(ValidationError):
        TrainingBudget(**{field: 0})


def test_primary_metric_is_mae():
    with pytest.raises(ValidationError):
        make_configuration(
            primary_metric=RegressionMetric.RMSE,
        )


def test_duplicate_secondary_metrics():
    with pytest.raises(ValidationError):
        make_configuration(
            secondary_metrics=(
                RegressionMetric.RMSE,
                RegressionMetric.RMSE,
            ),
        )


def test_primary_metric_cannot_be_secondary():
    with pytest.raises(ValidationError):
        make_configuration(
            secondary_metrics=(
                RegressionMetric.MAE,
            ),
        )


def test_non_regression_configuration_rejected():
    with pytest.raises(ValidationError):
        make_configuration(
            task_type=TaskType.CLASSIFICATION,
        )


def test_confirmed_target_contract():
    config = make_configuration()

    config.validate_contract(
        make_contract(confirmed=True)
    )


def test_unconfirmed_target_rejected():
    config = make_configuration()

    with pytest.raises(
        ValueError,
        match="confirmed target",
    ):
        config.validate_contract(
            make_contract(confirmed=False)
        )


def test_historical_rate_requires_definition():
    config = make_configuration(
        candidates=(
            make_baseline(),
            make_challenger(),
            CandidateConfiguration(
                family=ModelFamily.HISTORICAL_RATE,
                role=CandidateRole.BASELINE,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="requires a rate definition",
    ):
        config.validate_contract(
            make_contract()
        )


def test_development_split_excludes_test_partition():
    manifest = make_manifest()

    development = DevelopmentSplit.from_manifest(
        manifest
    )

    assert development.train == manifest.train
    assert development.validation == manifest.validation

    payload = development.model_dump()

    assert "test" not in payload
    assert "excluded_record_ids" not in payload

    development_ids = (
        set(development.train.record_ids)
        | set(development.validation.record_ids)
    )

    assert not (
        development_ids
        & set(manifest.test.record_ids)
    )


def test_configuration_is_frozen():
    config = make_configuration()

    with pytest.raises(ValidationError):
        config.random_seed = 100


def test_configuration_serialization():
    config = make_configuration()

    restored = TrainingConfiguration.model_validate_json(
        config.model_dump_json()
    )

    assert restored == config