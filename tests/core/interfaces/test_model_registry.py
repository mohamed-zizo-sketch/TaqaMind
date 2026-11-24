"""Tests for model registry and lifecycle contracts."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    ModelFeatureSpecification,
    ModelRegistry,
    ModelRegistryEntry,
    ModelScope,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PredictionModelIdentity,
    PromotionComparison,
    validate_model_transition,
)
from tests.core.interfaces.test_evaluation import make_evaluation
from tests.core.interfaces.test_readiness import make_contract


def make_version(evaluation=None, **overrides):
    if evaluation is None:
        evaluation = make_evaluation()

    selection = evaluation.selection
    run = selection.training_run
    development = run.development_split
    context = development.context

    candidate = run.configuration.get_candidate(
        selection.selected_candidate_id
    )

    selected_result = run.best_challenger

    identity = PredictionModelIdentity(
        model_version_id=uuid4(),
        factory_id=context.factory_id,
        domain=context.domain,
        process=context.process,
        task_id=context.task_id,
        dataset_id=context.dataset_id,
        contract_version=context.contract_version,
        task_version=context.task_version,
        target_version=context.target_version,
    )

    data = {
        "identity": identity,
        "version_number": 1,
        "training_run_id": run.training_run_id,
        "selection_id": selection.selection_id,
        "training_dataset_revision_id": development.dataset_revision_id,
        "split_id": development.split_id,
        "candidate": candidate,
        "artifact": selected_result.model_reference,
        "feature_specification": ModelFeatureSpecification(
            feature_names=("planned_quantity_kg",),
            target_name="actual_waste_kg",
            target_unit="kg",
        ),
        "contract_sha256": "a" * 64,
        "implementation_sha256": "b" * 64,
    }

    data.update(overrides)

    return ModelVersion(**data)


def make_candidate(evaluation=None):
    return ModelRegistryEntry(
        version=make_version(evaluation),
    )


def make_validated():
    evaluation = make_evaluation()

    candidate = make_candidate(evaluation)

    return candidate.apply_evaluation(evaluation)


def make_model_pair(target_status=ModelStatus.VALIDATED):
    """Synthetic entries for lifecycle state validation only."""

    validated = make_validated()

    active = ModelRegistryEntry(
        version=validated.version,
        status=ModelStatus.ACTIVE,
        evaluation_id=validated.evaluation_id,
    )

    identity_payload = active.version.identity.model_dump()
    identity_payload["model_version_id"] = uuid4()

    identity = PredictionModelIdentity.model_validate(
        identity_payload
    )

    version_payload = active.version.model_dump()

    version_payload.update(
        {
            "identity": identity.model_dump(),
            "version_number": 2,
            "previous_model_version_id": (
                active.version.model_version_id
            ),
        }
    )

    target = ModelRegistryEntry(
        version=ModelVersion.model_validate(version_payload),
        status=target_status,
        evaluation_id=(
            None
            if target_status == ModelStatus.CANDIDATE
            else uuid4()
        ),
    )

    return active, target


def make_comparison(active, target, **overrides):
    data = {
        "scope": active.version.scope,
        "active_model_version_id": active.version.model_version_id,
        "challenger_model_version_id": target.version.model_version_id,
        "evaluation_dataset_revision_id": uuid4(),
        "evaluation_record_ids": tuple(
            uuid4() for _ in range(10)
        ),
        "active_mae": 10.0,
        "challenger_mae": 7.0,
        "minimum_improvement_pct": 10.0,
    }

    data.update(overrides)

    return PromotionComparison(**data)


def make_promotion_request(active, target, **overrides):
    data = {
        "scope": active.version.scope,
        "action": ModelTransitionAction.PROMOTION,
        "target_model_version_id": target.version.model_version_id,
        "expected_active_model_version_id": (
            active.version.model_version_id
        ),
        "requested_by": "synthetic_admin",
        "reason": "Synthetic model promotion.",
        "comparison": make_comparison(active, target),
    }

    data.update(overrides)

    return ModelTransitionRequest(**data)


def test_valid_feature_specification():
    specification = ModelFeatureSpecification(
        feature_names=("planned_quantity_kg",),
        target_name="actual_waste_kg",
        target_unit="kg",
    )

    specification.validate_contract(make_contract())


def test_target_cannot_be_predictor():
    with pytest.raises(ValidationError):
        ModelFeatureSpecification(
            feature_names=("actual_waste_kg",),
            target_name="actual_waste_kg",
        )


def test_duplicate_predictors_rejected():
    with pytest.raises(ValidationError):
        ModelFeatureSpecification(
            feature_names=(
                "planned_quantity_kg",
                "planned_quantity_kg",
            ),
            target_name="actual_waste_kg",
        )


def test_unknown_predictor_rejected():
    specification = ModelFeatureSpecification(
        feature_names=("unknown_feature",),
        target_name="actual_waste_kg",
        target_unit="kg",
    )

    with pytest.raises(ValueError, match="unapproved predictors"):
        specification.validate_contract(make_contract())


def test_valid_model_scope():
    version = make_version()

    version.scope.validate_identity(version.identity)


def test_cross_factory_scope_rejected():
    version = make_version()

    scope = ModelScope(
        factory_id=uuid4(),
        domain=version.identity.domain,
        process=version.identity.process,
        task_id=version.identity.task_id,
    )

    with pytest.raises(ValueError, match="scope mismatch"):
        scope.validate_identity(version.identity)


def test_model_matches_frozen_selection():
    evaluation = make_evaluation()

    version = make_version(evaluation)

    version.validate_selection(evaluation.selection)


def test_wrong_selection_rejected():
    evaluation = make_evaluation()

    version = make_version(
        evaluation,
        selection_id=uuid4(),
    )

    with pytest.raises(ValueError, match="selection mismatch"):
        version.validate_selection(evaluation.selection)


def test_candidate_initial_status():
    candidate = make_candidate()

    assert candidate.status == ModelStatus.CANDIDATE
    assert candidate.evaluation_id is None


def test_successful_evaluation_validates_model():
    evaluation = make_evaluation()

    candidate = make_candidate(evaluation)

    validated = candidate.apply_evaluation(evaluation)

    assert validated.status == ModelStatus.VALIDATED
    assert validated.evaluation_id == evaluation.evaluation_id
    assert candidate.status == ModelStatus.CANDIDATE


def test_failed_evaluation_rejects_model():
    evaluation = make_evaluation(
        baseline_mae=10.0,
        selected_mae=15.0,
    )

    candidate = make_candidate(evaluation)

    rejected = candidate.apply_evaluation(evaluation)

    assert rejected.status == ModelStatus.REJECTED


def test_active_model_requires_evaluation():
    with pytest.raises(ValidationError):
        ModelRegistryEntry(
            version=make_version(),
            status=ModelStatus.ACTIVE,
        )


def test_final_evaluation_cannot_be_applied_twice():
    evaluation = make_evaluation()

    candidate = make_candidate(evaluation)

    validated = candidate.apply_evaluation(evaluation)

    with pytest.raises(ValueError, match="Only candidate"):
        validated.apply_evaluation(evaluation)


def test_valid_first_activation():
    target = make_validated()

    request = ModelTransitionRequest(
        scope=target.version.scope,
        action=ModelTransitionAction.FIRST_ACTIVATION,
        target_model_version_id=target.version.model_version_id,
        requested_by="system",
        reason="First validated model.",
    )

    validate_model_transition(
        request=request,
        target=target,
        active=None,
    )


def test_first_activation_rejects_existing_active_model():
    active, target = make_model_pair()

    request = ModelTransitionRequest(
        scope=target.version.scope,
        action=ModelTransitionAction.FIRST_ACTIVATION,
        target_model_version_id=target.version.model_version_id,
        requested_by="system",
        reason="First activation.",
    )

    with pytest.raises(ValueError, match="Active model changed"):
        validate_model_transition(
            request=request,
            target=target,
            active=active,
        )


def test_valid_model_comparison():
    active, target = make_model_pair()

    comparison = make_comparison(active, target)

    assert comparison.passes
    assert comparison.improvement_pct == pytest.approx(30.0)


def test_valid_promotion():
    active, target = make_model_pair()

    request = make_promotion_request(active, target)

    validate_model_transition(
        request=request,
        target=target,
        active=active,
    )


def test_promotion_requires_comparison():
    active, target = make_model_pair()

    with pytest.raises(
        ValidationError,
        match="requires a model comparison",
    ):
        make_promotion_request(
            active,
            target,
            comparison=None,
        )


def test_insufficient_improvement_blocks_promotion():
    active, target = make_model_pair()

    comparison = make_comparison(
        active,
        target,
        active_mae=10.0,
        challenger_mae=9.8,
    )

    assert not comparison.passes

    with pytest.raises(ValidationError):
        make_promotion_request(
            active,
            target,
            comparison=comparison,
        )


def test_zero_active_mae_blocks_relative_promotion():
    active, target = make_model_pair()

    comparison = make_comparison(
        active,
        target,
        active_mae=0,
        challenger_mae=0,
    )

    assert comparison.improvement_pct is None
    assert not comparison.passes


@pytest.mark.parametrize(
    "status",
    [
        ModelStatus.CANDIDATE,
        ModelStatus.REJECTED,
        ModelStatus.ARCHIVED,
    ],
)
def test_invalid_promotion_target_status(status):
    active, target = make_model_pair(
        target_status=status,
    )

    request = make_promotion_request(active, target)

    with pytest.raises(
        ValueError,
        match="invalid lifecycle status",
    ):
        validate_model_transition(
            request=request,
            target=target,
            active=active,
        )


def test_valid_rollback():
    active, archived = make_model_pair(
        target_status=ModelStatus.ARCHIVED,
    )

    request = ModelTransitionRequest(
        scope=active.version.scope,
        action=ModelTransitionAction.ROLLBACK,
        target_model_version_id=archived.version.model_version_id,
        expected_active_model_version_id=active.version.model_version_id,
        requested_by="synthetic_admin",
        reason="Synthetic rollback.",
    )

    validate_model_transition(
        request=request,
        target=archived,
        active=active,
    )


def test_model_version_serialization():
    version = make_version()

    restored = ModelVersion.model_validate_json(
        version.model_dump_json()
    )

    assert restored == version


def test_registry_entry_serialization():
    entry = make_validated()

    restored = ModelRegistryEntry.model_validate_json(
        entry.model_dump_json()
    )

    assert restored == entry


def test_model_version_is_immutable():
    version = make_version()

    with pytest.raises(ValidationError):
        version.version_number = 10


def test_empty_object_is_not_registry():
    assert not isinstance(object(), ModelRegistry)