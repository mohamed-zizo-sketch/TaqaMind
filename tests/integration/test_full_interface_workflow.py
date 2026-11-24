"""Cross-component integration tests for the shared pipeline."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.interfaces import (
    AcceptancePolicy,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    DevelopmentDataAccess,
    EvaluationDecision,
    EvaluationReservation,
    FinalEvaluationResult,
    FrozenModelSelection,
    LeakageCheck,
    ModelRegistryEntry,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    PipelineContext,
    PredictionBatchResult,
    PredictionBatchStatus,
    PredictionStatus,
    ProtectedTestDataAccess,
    RecordPrediction,
    RecordUsageRole,
    StageTiming,
    TrainingAuthorization,
    TrainingIntent,
    TrainingMode,
    TrainingReuseKey,
    TrainingRunResult,
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
    deserialize_contract,
    find_previously_exposed_test_records,
    serialize_contract,
    validate_active_model_access,
    validate_model_transition,
    validate_prediction_request,
    validate_training_request,
)
from taqamind.core.interfaces import (
    TestMetricResult as ProtectedTestMetrics,
)
from tests.core.interfaces.test_candidates import (
    make_successful_result,
)
from tests.core.interfaces.test_data_access import (
    make_partition_reference,
)
from tests.core.interfaces.test_evaluation import make_metrics
from tests.core.interfaces.test_leakage import make_validation
from tests.core.interfaces.test_model_registry import make_version
from tests.core.interfaces.test_training import make_configuration
from tests.core.interfaces.test_training_registry import (
    make_membership,
    make_scope,
    make_usage,
)


def build_training_workflow(
    *,
    test_baseline_mae=12.0,
    test_model_mae=8.0,
):
    """Build a consistent synthetic ML workflow."""

    leakage = make_validation()

    readiness = leakage.readiness
    contract = readiness.contract

    manifest = leakage.manifest
    context = manifest.context

    authorization = TrainingAuthorization.from_validation(
        leakage
    )

    development = authorization.development_split

    configuration = make_configuration()

    policy = AcceptancePolicy(
        minimum_improvement_pct=10.0,
        justification="Synthetic integration acceptance policy.",
        created_at=(
            context.created_at - timedelta(seconds=1)
        ),
    )

    intent = TrainingIntent(
        context=context,
        mode=TrainingMode.INITIAL,
        requested_by="synthetic_admin",
        reason="Initial integration test training.",
    )

    membership = make_membership(manifest)

    membership.validate_artifact(
        readiness.result.input_artifact
    )

    split_hash = calculate_split_fingerprint(
        manifest,
        membership,
    )

    configuration_hash = (
        calculate_training_configuration_fingerprint(
            configuration,
            policy,
        )
    )

    reuse_key = TrainingReuseKey.from_intent(
        intent=intent,
        dataset_content_sha256=(
            readiness.result.input_artifact.reference.content_sha256
        ),
        contract_sha256="a" * 64,
        split_sha256=split_hash,
        training_configuration_sha256=configuration_hash,
        implementation_sha256="b" * 64,
    )

    validate_training_request(
        authorization=authorization,
        contract=contract,
        configuration=configuration,
        policy=policy,
    )

    train_reference = make_partition_reference(
        context=context,
        split_id=manifest.split_id,
        dataset_revision_id=manifest.dataset_revision_id,
        partition=manifest.train,
    )

    validation_reference = make_partition_reference(
        context=context,
        split_id=manifest.split_id,
        dataset_revision_id=manifest.dataset_revision_id,
        partition=manifest.validation,
    )

    development_access = DevelopmentDataAccess(
        authorization=authorization,
        train=train_reference,
        validation=validation_reference,
    )

    development_access.validate_contract(contract)

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

    reservation = EvaluationReservation.from_selection(
        scope=membership.scope,
        selection=selection,
        manifest=manifest,
    )

    test_reference = make_partition_reference(
        context=context,
        split_id=manifest.split_id,
        dataset_revision_id=manifest.dataset_revision_id,
        partition=manifest.test,
    )

    test_access = ProtectedTestDataAccess(
        reservation=reservation,
        selection=selection,
        test=test_reference,
    )

    test_access.validate_contract(contract)

    assert baseline.model_reference is not None
    assert challenger.model_reference is not None

    evaluation = FinalEvaluationResult(
        selection=selection,
        split_manifest=manifest,
        baseline_result=ProtectedTestMetrics(
            candidate_id=baseline.candidate_id,
            model_artifact_id=(
                baseline.model_reference.artifact_id
            ),
            record_ids=manifest.test.record_ids,
            metrics=make_metrics(test_baseline_mae),
        ),
        selected_result=ProtectedTestMetrics(
            candidate_id=challenger.candidate_id,
            model_artifact_id=(
                challenger.model_reference.artifact_id
            ),
            record_ids=manifest.test.record_ids,
            metrics=make_metrics(test_model_mae),
        ),
        timing=StageTiming(
            started_at=reservation.reserved_at,
            finished_at=datetime.now(UTC),
        ),
    )

    candidate = ModelRegistryEntry(
        version=make_version(evaluation),
    )

    candidate.version.feature_specification.validate_contract(
        contract
    )

    evaluated_model = candidate.apply_evaluation(
        evaluation
    )

    return {
        "contract": contract,
        "leakage": leakage,
        "manifest": manifest,
        "membership": membership,
        "intent": intent,
        "reuse_key": reuse_key,
        "development_access": development_access,
        "training_run": run,
        "selection": selection,
        "reservation": reservation,
        "test_access": test_access,
        "evaluation": evaluation,
        "candidate": candidate,
        "evaluated_model": evaluated_model,
    }


def simulate_first_activation(entry):
    """Simulate a permitted state transition without persistence."""

    request = ModelTransitionRequest(
        scope=entry.version.scope,
        action=ModelTransitionAction.FIRST_ACTIVATION,
        target_model_version_id=(
            entry.version.model_version_id
        ),
        requested_by="system",
        reason="Synthetic first activation.",
    )

    validate_model_transition(
        request=request,
        target=entry,
        active=None,
    )

    return ModelRegistryEntry(
        version=entry.version,
        status=ModelStatus.ACTIVE,
        evaluation_id=entry.evaluation_id,
    )


def make_prediction_input(
    *,
    contract,
    entry,
    row_count,
):
    """Create synthetic prediction metadata for an existing model."""

    identity = entry.version.identity

    context = PipelineContext(
        factory_id=identity.factory_id,
        dataset_revision_id=uuid4(),
        domain=identity.domain,
        process=identity.process,
        task_id=identity.task_id,
        dataset_id=identity.dataset_id,
        contract_version=identity.contract_version,
        task_version=identity.task_version,
        target_version=identity.target_version,
        purpose=DatasetPurpose.PREDICTION,
        model_version_id=identity.model_version_id,
    )

    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256="c" * 64,
    )

    columns = (
        DatasetColumnSchema(
            name="planned_quantity_kg",
            physical_dtype="float64",
        ),
    )

    for state in (
        DatasetState.INGESTED,
        DatasetState.MAPPED,
        DatasetState.CLEANED,
        DatasetState.ENGINEERED,
    ):
        metadata = metadata.derive(
            context=context,
            state=state,
            row_count=row_count,
            columns=columns,
            operation="synthetic_processing",
            implementation_version="1.0.0",
            description="Synthetic prediction processing.",
        )

    artifact = DatasetArtifact(
        metadata=metadata,
        reference=ArtifactReference(
            factory_id=context.factory_id,
            format=ArtifactFormat.CSV,
            content_sha256="d" * 64,
            size_bytes=1024,
        ),
    )

    validate_active_model_access(
        context=context,
        entry=entry,
        contract=contract,
    )

    validate_prediction_request(
        context=context,
        artifact=artifact,
        contract=contract,
        model=identity,
    )

    return context, artifact


def test_complete_training_contract_workflow():
    workflow = build_training_workflow()

    assert workflow["leakage"].eligible_for_training

    assert workflow[
        "training_run"
    ].eligible_for_final_evaluation

    assert workflow["evaluation"].decision == (
        EvaluationDecision.ACCEPTED
    )

    assert workflow["candidate"].status == (
        ModelStatus.CANDIDATE
    )

    assert workflow["evaluated_model"].status == (
        ModelStatus.VALIDATED
    )

    active = simulate_first_activation(
        workflow["evaluated_model"]
    )

    assert active.status == ModelStatus.ACTIVE

    assert (
        active.version.model_version_id
        == workflow["candidate"].version.model_version_id
    )


def test_development_access_excludes_protected_test():
    workflow = build_training_workflow()

    access = workflow["development_access"]

    test = workflow["test_access"].test

    development_ids = (
        set(access.train.record_ids)
        | set(access.validation.record_ids)
    )

    assert not development_ids & set(test.record_ids)

    assert not hasattr(access, "test")


def test_same_model_identity_across_prediction_executions():
    workflow = build_training_workflow()

    active = simulate_first_activation(
        workflow["evaluated_model"]
    )

    contract = workflow["contract"]

    contexts = []
    model_ids = []

    for row_count in (2, 5, 10):
        context, artifact = make_prediction_input(
            contract=contract,
            entry=active,
            row_count=row_count,
        )

        batch = PredictionBatchResult(
            context=context,
            model=active.version.identity,
            contract=contract,
            input_artifact=artifact,
            predictions=tuple(
                RecordPrediction(
                    record_id=uuid4(),
                    status=PredictionStatus.SUCCEEDED,
                    predicted_value=30.0,
                )
                for _ in range(row_count)
            ),
            timing=StageTiming(
                started_at=context.created_at,
                finished_at=datetime.now(UTC),
            ),
        )

        assert batch.status == (
            PredictionBatchStatus.SUCCEEDED
        )

        assert batch.successful_count == row_count

        contexts.append(context.execution_id)
        model_ids.append(batch.model.model_version_id)

    assert len(set(contexts)) == 3

    assert len(set(model_ids)) == 1

    assert active.version.version_number == 1


def test_rejected_model_cannot_be_activated():
    workflow = build_training_workflow(
        test_baseline_mae=10.0,
        test_model_mae=15.0,
    )

    rejected = workflow["evaluated_model"]

    assert rejected.status == ModelStatus.REJECTED

    with pytest.raises(
        ValueError,
        match="invalid lifecycle status",
    ):
        simulate_first_activation(rejected)


def test_failed_leakage_stops_training_authorization():
    leakage = make_validation(
        failed_check=LeakageCheck.HISTORICAL_USAGE,
    )

    with pytest.raises(
        ValueError,
        match="does not permit production training",
    ):
        TrainingAuthorization.from_validation(
            leakage
        )


def test_cross_factory_prediction_is_rejected():
    workflow = build_training_workflow()

    active = simulate_first_activation(
        workflow["evaluated_model"]
    )

    contract = workflow["contract"]

    context, _ = make_prediction_input(
        contract=contract,
        entry=active,
        row_count=2,
    )

    payload = context.model_dump()
    payload["factory_id"] = uuid4()

    incompatible = PipelineContext.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="factory_id",
    ):
        validate_active_model_access(
            context=incompatible,
            entry=active,
            contract=contract,
        )


def test_corrected_record_is_still_historically_exposed():
    workflow = build_training_workflow()

    manifest = workflow["manifest"]

    record_id = manifest.test.record_ids[0]

    usage = make_usage(
        make_scope(manifest),
        record_id,
        role=RecordUsageRole.TRAIN,
        record_version_id=uuid4(),
    )

    exposed = find_previously_exposed_test_records(
        manifest=manifest,
        scope=make_scope(manifest),
        history=(usage,),
    )

    assert exposed == (record_id,)


def test_training_reuse_key_survives_serialization():
    workflow = build_training_workflow()

    original = workflow["reuse_key"]

    restored = deserialize_contract(
        TrainingReuseKey,
        serialize_contract(original),
    )

    assert restored.fingerprint == original.fingerprint


def test_changed_dataset_invalidates_training_reuse():
    workflow = build_training_workflow()

    original = workflow["reuse_key"]

    payload = original.model_dump()
    payload["dataset_content_sha256"] = "f" * 64

    changed = TrainingReuseKey.model_validate(
        payload
    )

    assert changed.fingerprint != original.fingerprint


def test_entire_workflow_contracts_round_trip():
    workflow = build_training_workflow()

    for name, original in workflow.items():
        restored = deserialize_contract(
            type(original),
            serialize_contract(original),
        )

        assert restored == original, name