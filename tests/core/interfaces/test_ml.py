"""Tests for shared machine learning protocols."""

from datetime import timedelta
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetContract
from taqamind.core.interfaces import (
    AcceptancePolicy,
    DatasetSplitter,
    FinalModelEvaluator,
    LeakageValidator,
    ModelTrainer,
    PredictionEngine,
    PredictionModelIdentity,
    TrainingAuthorization,
    TrainingReadinessChecker,
    validate_prediction_request,
    validate_training_request,
)
from tests.core.interfaces.test_leakage import make_validation
from tests.core.interfaces.test_prediction import make_prediction_batch
from tests.core.interfaces.test_training import make_configuration


class SyntheticMLEngine:
    """Structural protocol test double."""

    def assess(self, *, context, artifact, contract):
        raise NotImplementedError

    def split(self, *, readiness, configuration):
        raise NotImplementedError

    def validate(self, *, readiness, manifest):
        raise NotImplementedError

    def train(
        self,
        *,
        authorization,
        contract,
        configuration,
        policy,
    ):
        raise NotImplementedError

    def evaluate(self, *, selection, manifest):
        raise NotImplementedError

    def predict(
        self,
        *,
        context,
        artifact,
        contract,
        model,
    ):
        raise NotImplementedError


@pytest.mark.parametrize(
    "protocol",
    [
        TrainingReadinessChecker,
        DatasetSplitter,
        LeakageValidator,
        ModelTrainer,
        FinalModelEvaluator,
        PredictionEngine,
    ],
)
def test_protocol_compatibility(protocol):
    engine = SyntheticMLEngine()

    assert isinstance(engine, protocol)


@pytest.mark.parametrize(
    "protocol",
    [
        TrainingReadinessChecker,
        DatasetSplitter,
        LeakageValidator,
        ModelTrainer,
        FinalModelEvaluator,
        PredictionEngine,
    ],
)
def test_empty_object_is_not_compatible(protocol):
    assert not isinstance(object(), protocol)


def make_training_request():
    validation = make_validation()

    authorization = TrainingAuthorization.from_validation(
        validation
    )

    context = authorization.development_split.context

    policy = AcceptancePolicy(
        minimum_improvement_pct=10,
        justification="Synthetic acceptance policy.",
        created_at=(
            context.created_at - timedelta(seconds=1)
        ),
    )

    return {
        "authorization": authorization,
        "contract": validation.readiness.contract,
        "configuration": make_configuration(),
        "policy": policy,
    }


def test_valid_training_request():
    validate_training_request(
        **make_training_request()
    )


def test_training_contract_mismatch():
    request = make_training_request()

    payload = request["contract"].model_dump()
    payload["version"] = "2.0.0"

    request["contract"] = DatasetContract.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="version",
    ):
        validate_training_request(**request)


def test_acceptance_policy_must_precede_training():
    request = make_training_request()

    context = request[
        "authorization"
    ].development_split.context

    request["policy"] = AcceptancePolicy(
        minimum_improvement_pct=10,
        justification="Synthetic late policy.",
        created_at=(
            context.created_at + timedelta(seconds=1)
        ),
    )

    with pytest.raises(
        ValueError,
        match="before training",
    ):
        validate_training_request(**request)


def test_valid_prediction_request():
    batch = make_prediction_batch()

    validate_prediction_request(
        context=batch.context,
        artifact=batch.input_artifact,
        contract=batch.contract,
        model=batch.model,
    )


def test_prediction_rejects_different_factory_model():
    batch = make_prediction_batch()

    payload = batch.model.model_dump()
    payload["factory_id"] = uuid4()

    other_model = PredictionModelIdentity.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="factory_id",
    ):
        validate_prediction_request(
            context=batch.context,
            artifact=batch.input_artifact,
            contract=batch.contract,
            model=other_model,
        )


def test_prediction_rejects_different_model_version():
    batch = make_prediction_batch()

    payload = batch.model.model_dump()
    payload["model_version_id"] = uuid4()

    other_model = PredictionModelIdentity.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="model_version_id",
    ):
        validate_prediction_request(
            context=batch.context,
            artifact=batch.input_artifact,
            contract=batch.contract,
            model=other_model,
        )