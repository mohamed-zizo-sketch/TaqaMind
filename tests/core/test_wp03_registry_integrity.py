"""WP03 regression tests — Registry Integrity, atomic rollback, and observation persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from taqamind.core.automl.training_registry import LocalTrainingRegistry
from taqamind.core.contracts.dataset import DatasetContract, PredictionContext
from taqamind.core.contracts.enums import (
    ConfirmationStatus,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
    PipelineStage,
    TargetKind,
    TaskType,
)
from taqamind.core.contracts.feature import FeatureConstraints, FeatureContract
from taqamind.core.contracts.target import TargetConfirmation, TargetContract, TargetDefinition
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    CandidateConfiguration,
    CandidateModelReference,
    CandidateResult,
    CandidateRole,
    CandidateStatus,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMembershipManifest,
    DatasetMetadata,
    DatasetRecordMembership,
    DatasetState,
    DevelopmentSplit,
    ExecutionError,
    FingerprintField,
    ModelFeatureSpecification,
    ModelScope,
    ModelStatus,
    ModelTransitionAction,
    ModelTransitionRequest,
    ModelVersion,
    PipelineContext,
    PredictionModelIdentity,
    RecordIdentity,
    RecordIdentityStatus,
    RecordScope,
    RecordUsage,
    RecordUsageRole,
    RecordVersion,
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryScopeError,
    RegressionMetric,
    SplitConfiguration,
    SplitManifest,
    SplitPartition,
    SplitRole,
    SplitStrategy,
    StageTiming,
    TrainingConfiguration,
    TrainingMode,
    TrainingReuseKey,
    TrainingRunResult,
    TransformationRecord,
    calculate_bytes_sha256,
    calculate_record_fingerprint,
)
from taqamind.core.interfaces.lineage import (
    PredictionIdentityStatus,
    PredictionObservation,
    PredictionObservationBatch,
)
from taqamind.core.lineage import LocalRecordLineageRegistry, RecordMembershipBuilder
from taqamind.core.registry import (
    LocalDatasetArtifactStorage,
    LocalModelArtifactStorage,
    LocalModelRegistry,
)
from taqamind.core.splitting.split_registry import LocalRecordUsageRegistry


def _scope(factory_id=None) -> RecordScope:
    return RecordScope(
        factory_id=factory_id or uuid4(),
        domain="textile",
        process="weaving",
        dataset_id="wp03_ds",
        identity_namespace="production_record",
        identity_version="2.0.0",
    )


def _sha(label: str) -> str:
    return calculate_bytes_sha256(label.encode("utf-8"))


class TestLineageBatchAtomicityAndObservations:
    """WP03: All-or-nothing batch registration and persistent prediction observations."""

    def test_register_batch_rolls_back_all_files_and_index_on_failure(self, tmp_path: Path):
        lin = LocalRecordLineageRegistry(tmp_path / "reg")
        scope = _scope()
        rev_id = uuid4()

        id1 = RecordIdentity(
            scope=scope,
            business_key_sha256=_sha("bk-1"),
            status=RecordIdentityStatus.CONFIRMED,
        )
        id2 = RecordIdentity(
            scope=scope,
            business_key_sha256=_sha("bk-2"),
            status=RecordIdentityStatus.CONFIRMED,
        )
        ver1 = RecordVersion(
            identity=id1,
            content_sha256=_sha("c-1"),
            content_fingerprint_version="2.0.0",
            source_dataset_revision_id=rev_id,
        )
        # Missing identity for ver_bad will raise RegistryNotFoundError mid-batch.
        ghost_id = RecordIdentity(
            scope=scope,
            business_key_sha256=_sha("ghost"),
            status=RecordIdentityStatus.CONFIRMED,
        )
        ver_bad = RecordVersion(
            identity=ghost_id,
            content_sha256=_sha("c-bad"),
            content_fingerprint_version="2.0.0",
            source_dataset_revision_id=rev_id,
        )
        manifest = DatasetMembershipManifest(
            dataset_revision_id=rev_id,
            scope=scope,
            memberships=(
                DatasetRecordMembership(
                    dataset_revision_id=rev_id,
                    scope=scope,
                    row_position=0,
                    record_id=id1.record_id,
                    record_version_id=ver1.record_version_id,
                ),
            ),
        )

        with pytest.raises(RegistryNotFoundError):
            lin.register_batch(
                scope=scope,
                identities=(id1, id2),
                versions=(ver1, ver_bad),
                manifest=manifest,
            )

        scope_dir = lin._scope_dir(scope)
        assert list((scope_dir / "identities").glob("*.json")) == []
        assert list((scope_dir / "versions").glob("*.json")) == []
        assert not (scope_dir / "memberships" / f"{rev_id}.json").exists()
        assert lin._find_identities_by_business_key(scope, _sha("bk-1")) == []

    def test_prediction_observations_persist_isolated_from_historical_lineage(
        self, tmp_path: Path
    ):
        storage = LocalDatasetArtifactStorage(tmp_path / "blobs")
        lin = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin)
        factory_id = uuid4()
        scope = RecordScope(
            factory_id=factory_id,
            domain="textile",
            process="weaving",
            dataset_id="wp03_ds",
            identity_namespace="production_record",
            identity_version="1.0.0",
        )
        contract = DatasetContract(
            dataset_id="wp03_ds",
            version="1.0.0",
            domain="textile",
            process="weaving",
            description="WP03 contract",
            granularity="batch",
            features=(
                FeatureContract(
                    name="batch_id",
                    display_name="Batch ID",
                    description="Key",
                    data_type=DataType.STRING,
                    role=FeatureRole.IDENTIFIER,
                    availability=FeatureAvailability.PRE_PRODUCTION,
                    origin=FeatureOrigin.RECORDED,
                    missing_policy=MissingValuePolicy.REJECT,
                ),
                FeatureContract(
                    name="qty",
                    display_name="Qty",
                    description="Qty",
                    data_type=DataType.FLOAT,
                    role=FeatureRole.PREDICTOR,
                    availability=FeatureAvailability.PRE_PRODUCTION,
                    origin=FeatureOrigin.RECORDED,
                    missing_policy=MissingValuePolicy.REJECT,
                ),
            ),
            target=TargetContract(
                target_id="waste_target",
                feature=FeatureContract(
                    name="waste",
                    display_name="Waste",
                    description="Waste",
                    data_type=DataType.FLOAT,
                    role=FeatureRole.TARGET,
                    availability=FeatureAvailability.POST_PRODUCTION,
                    origin=FeatureOrigin.MEASURED,
                    unit="kg",
                    constraints=FeatureConstraints(minimum=0.0),
                    missing_policy=MissingValuePolicy.REJECT,
                ),
                task_type=TaskType.REGRESSION,
                target_kind=TargetKind.QUANTITY,
                definition=TargetDefinition(definition="Waste", granularity="batch"),
                confirmation=TargetConfirmation(
                    status=ConfirmationStatus.CONFIRMED,
                    confirmed_by="wp03_test",
                    confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ),
            prediction_context=PredictionContext(description="Batch pred"),
            business_keys=("batch_id",),
        )
        csv_bytes = b"batch_id,qty\nB1,10.0\n"
        ref = ArtifactReference(
            factory_id=factory_id,
            format=ArtifactFormat.CSV,
            content_sha256=calculate_bytes_sha256(csv_bytes),
            size_bytes=len(csv_bytes),
        )
        storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))
        root_id = uuid4()
        rev_id = uuid4()
        artifact = DatasetArtifact(
            metadata=DatasetMetadata(
                revision_id=rev_id,
                root_revision_id=root_id,
                factory_id=factory_id,
                domain="textile",
                process="weaving",
                dataset_id="wp03_ds",
                contract_version="1.0.0",
                task_id="waste_target",
                task_version="1.0.0",
                target_version="1.0.0",
                source_file_sha256=ref.content_sha256,
                state=DatasetState.MAPPED,
                row_count=1,
                columns=(
                    DatasetColumnSchema(name="batch_id", physical_dtype="object"),
                    DatasetColumnSchema(name="qty", physical_dtype="float64"),
                ),
                transformation=TransformationRecord(
                    execution_id=uuid4(),
                    parent_revision_id=root_id,
                    stage=PipelineStage.COLUMN_MAPPING,
                    operation="column_mapping",
                    implementation_version="1.0.0",
                    description="Mapped",
                ),
            ),
            reference=ref,
        )
        ctx = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=rev_id,
            domain="textile",
            process="weaving",
            dataset_id="wp03_ds",
            contract_version="1.0.0",
            task_id="waste_target",
            task_version="1.0.0",
            target_version="1.0.0",
            purpose=DatasetPurpose.PREDICTION,
            model_version_id=uuid4(),
        )

        batch = builder.resolve_prediction_observations(
            scope=scope, artifact=artifact, contract=contract, context=ctx
        )
        loaded = lin.get_prediction_observations(scope=scope, dataset_revision_id=rev_id)
        assert loaded == batch
        assert loaded.observations[0].resolution_status == (
            PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE
        )
        # Historical lineage must remain untouched.
        scope_dir = lin._scope_dir(scope)
        assert not (scope_dir / "identities").exists()
        assert not (scope_dir / "versions").exists()
        assert not (scope_dir / "memberships").exists()

        # Conflicting observation batch for same revision raises RegistryConflictError.
        conflicting = PredictionObservationBatch(
            scope=scope,
            prediction_dataset_revision_id=rev_id,
            observations=(
                PredictionObservation(
                    scope=scope,
                    prediction_dataset_revision_id=rev_id,
                    row_position=0,
                    business_key_sha256=calculate_record_fingerprint(
                        (FingerprintField(name="batch_id", value="OTHER"),),
                        namespace="business_key",
                        version="2.0.0",
                    ),
                    prediction_input_sha256=_sha("other-input"),
                    resolution_status=PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE,
                    known_record_id=None,
                ),
            ),
        )
        with pytest.raises(RegistryConflictError):
            lin.register_prediction_observations(scope=scope, batch=conflicting)


class TestModelTransitionRollback:
    """WP03: ModelRegistry.transition rolls back archived active model on failure."""

    def test_failed_target_write_restores_previous_active_model(self, tmp_path: Path):
        model_storage = LocalModelArtifactStorage(tmp_path / "models")
        reg = LocalModelRegistry(root_dir=tmp_path / "reg", model_storage=model_storage)
        factory_id = uuid4()
        scope = ModelScope(
            factory_id=factory_id,
            domain="textile",
            process="weaving",
            task_id="waste_target",
        )

        def _make_version(num: int) -> ModelVersion:
            payload = f"bundle-{num}".encode()
            cand = CandidateConfiguration(
                family="ridge",
                role=CandidateRole.CHALLENGER,
            )
            ref = CandidateModelReference(
                factory_id=factory_id,
                execution_id=uuid4(),
                candidate_id=cand.candidate_id,
                content_sha256=calculate_bytes_sha256(payload),
                size_bytes=len(payload),
            )
            model_storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(payload))
            m_id = uuid4()
            return ModelVersion(
                identity=PredictionModelIdentity(
                    model_version_id=m_id,
                    factory_id=factory_id,
                    domain="textile",
                    process="weaving",
                    task_id="waste_target",
                    dataset_id="wp03_ds",
                    contract_version="1.0.0",
                    task_version="1.0.0",
                    target_version="1.0.0",
                ),
                version_number=num,
                training_run_id=uuid4(),
                selection_id=uuid4(),
                training_dataset_revision_id=uuid4(),
                split_id=uuid4(),
                candidate=cand,
                artifact=ref,
                feature_specification=ModelFeatureSpecification(
                    feature_names=("qty",),
                    target_name="waste",
                ),
                contract_sha256=_sha("contract"),
                implementation_sha256=_sha("impl"),
            )

        v1 = _make_version(1)
        v2 = _make_version(2)
        e1 = reg.register_candidate(scope=scope, version=v1)
        e2 = reg.register_candidate(scope=scope, version=v2)

        # Manually mark both validated and activate v1.
        from taqamind.core.interfaces.model_registry import ModelRegistryEntry, PromotionComparison
        from taqamind.core.registry._json_store import atomic_write_text, dump_json_model

        for e in (e1, e2):
            validated = ModelRegistryEntry(
                version=e.version,
                status=ModelStatus.VALIDATED,
                evaluation_id=uuid4(),
                updated_at=datetime.now(UTC),
            )
            atomic_write_text(
                reg._version_path(scope, e.version.model_version_id),
                dump_json_model(validated),
            )

        reg.transition(
            request=ModelTransitionRequest(
                scope=scope,
                action=ModelTransitionAction.FIRST_ACTIVATION,
                target_model_version_id=v1.model_version_id,
                expected_active_model_version_id=None,
                requested_by="admin",
                reason="Activate v1",
            )
        )

        target_path = reg._version_path(scope, v2.model_version_id)
        real_atomic = atomic_write_text
        call_count = {"n": 0}

        def _fail_on_target(path: Path, content: str) -> None:
            call_count["n"] += 1
            if path == target_path and call_count["n"] <= 2:
                raise OSError("Simulated disk failure during target activation")
            real_atomic(path, content)

        with (
            patch(
                "taqamind.core.registry.model_registry.atomic_write_text",
                side_effect=_fail_on_target,
            ),
            pytest.raises(OSError, match="Simulated disk failure"),
        ):
            reg.transition(
                request=ModelTransitionRequest(
                    scope=scope,
                    action=ModelTransitionAction.PROMOTION,
                    target_model_version_id=v2.model_version_id,
                    expected_active_model_version_id=v1.model_version_id,
                    requested_by="admin",
                    reason="Promote v2",
                    comparison=PromotionComparison(
                        scope=scope,
                        active_model_version_id=v1.model_version_id,
                        challenger_model_version_id=v2.model_version_id,
                        evaluation_dataset_revision_id=uuid4(),
                        evaluation_record_ids=(uuid4(),),
                        metric=RegressionMetric.MAE,
                        active_mae=0.5,
                        challenger_mae=0.1,
                        minimum_improvement_pct=1.0,
                    ),
                )
            )

        # v1 MUST still be ACTIVE and v2 MUST still be VALIDATED.
        active = reg.get_active(scope=scope)
        assert active is not None
        assert active.version.model_version_id == v1.model_version_id
        assert reg.get(scope=scope, model_version_id=v2.model_version_id).status == (
            ModelStatus.VALIDATED
        )


class TestUsageAndTrainingRegistryIntegrity:
    """WP03: RecordUsageRegistry atomicity and TrainingRegistry conflict protection."""

    def test_record_usage_is_all_or_nothing_on_scope_mismatch(self, tmp_path: Path):
        reg = LocalRecordUsageRegistry(tmp_path / "usage")
        scope_a = _scope()
        scope_b = _scope()
        u1 = RecordUsage(
            scope=scope_a,
            record_id=uuid4(),
            record_version_id=uuid4(),
            execution_id=uuid4(),
            split_id=uuid4(),
            role=RecordUsageRole.TRAIN,
        )
        u2_bad = RecordUsage(
            scope=scope_b,
            record_id=uuid4(),
            record_version_id=uuid4(),
            execution_id=uuid4(),
            split_id=uuid4(),
            role=RecordUsageRole.TRAIN,
        )

        with pytest.raises(RegistryScopeError):
            reg.record_usage(scope=scope_a, usages=(u1, u2_bad))

        assert reg.get_history(scope=scope_a, record_ids=(u1.record_id,)) == ()

    def test_record_usage_rejects_conflicting_usage_id(self, tmp_path: Path):
        reg = LocalRecordUsageRegistry(tmp_path / "usage")
        scope_a = _scope()
        uid = uuid4()
        u1 = RecordUsage(
            scope=scope_a,
            usage_id=uid,
            record_id=uuid4(),
            record_version_id=uuid4(),
            execution_id=uuid4(),
            split_id=uuid4(),
            role=RecordUsageRole.TRAIN,
        )
        u1_conflict = RecordUsage(
            scope=scope_a,
            usage_id=uid,
            record_id=uuid4(),
            record_version_id=uuid4(),
            execution_id=uuid4(),
            split_id=uuid4(),
            role=RecordUsageRole.PROTECTED_TEST,
        )
        reg.record_usage(scope=scope_a, usages=(u1,))
        with pytest.raises(RegistryConflictError):
            reg.record_usage(scope=scope_a, usages=(u1_conflict,))

    def test_training_registry_rejects_conflicting_reuse_key(self, tmp_path: Path):
        reg = LocalTrainingRegistry(tmp_path / "train")
        factory_id = uuid4()
        ctx = PipelineContext(
            factory_id=factory_id,
            dataset_revision_id=uuid4(),
            domain="textile",
            process="weaving",
            dataset_id="wp03_ds",
            contract_version="1.0.0",
            task_id="waste_target",
            task_version="1.0.0",
            target_version="1.0.0",
            purpose=DatasetPurpose.TRAINING,
        )
        r1, r2, r3 = uuid4(), uuid4(), uuid4()
        manifest = SplitManifest(
            context=ctx,
            dataset_revision_id=ctx.dataset_revision_id,
            configuration=SplitConfiguration(
                strategy=SplitStrategy.RANDOM,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
                justification="WP03 test split",
            ),
            train=SplitPartition(
                role=SplitRole.TRAIN,
                record_ids=(r1,),
            ),
            validation=SplitPartition(
                role=SplitRole.VALIDATION,
                record_ids=(r2,),
            ),
            test=SplitPartition(
                role=SplitRole.TEST,
                record_ids=(r3,),
            ),
        )
        cfg = TrainingConfiguration(
            random_seed=42,
            candidates=(
                CandidateConfiguration(
                    family="dummy_regressor",
                    role=CandidateRole.BASELINE,
                ),
                CandidateConfiguration(
                    family="ridge",
                    role=CandidateRole.CHALLENGER,
                ),
            ),
            primary_metric=RegressionMetric.MAE,
            justification="WP03 training reuse conflict test",
        )
        now = datetime.now(UTC)
        res1 = TrainingRunResult(
            development_split=DevelopmentSplit.from_manifest(manifest),
            configuration=cfg,
            results=(
                CandidateResult(
                    candidate_id=cfg.candidates[0].candidate_id,
                    status=CandidateStatus.FAILED,
                    timing=StageTiming(started_at=now, finished_at=now),
                    error=ExecutionError(code="TEST", message="Test failed baseline"),
                ),
                CandidateResult(
                    candidate_id=cfg.candidates[1].candidate_id,
                    status=CandidateStatus.FAILED,
                    timing=StageTiming(started_at=now, finished_at=now),
                    error=ExecutionError(code="TEST", message="Test failed challenger"),
                ),
            ),
        )
        res2 = res1.model_copy(update={"training_run_id": uuid4()})
        key = TrainingReuseKey(
            factory_id=factory_id,
            domain="textile",
            process="weaving",
            task_id="waste_target",
            dataset_id="wp03_ds",
            contract_version="1.0.0",
            task_version="1.0.0",
            target_version="1.0.0",
            mode=TrainingMode.INITIAL,
            dataset_content_sha256=_sha("ds"),
            contract_sha256=_sha("contract"),
            split_sha256=_sha("split"),
            training_configuration_sha256=_sha("cfg"),
            implementation_sha256=_sha("impl"),
        )
        reg.register_result(factory_id=factory_id, key=key, result=res1)
        with pytest.raises(RegistryConflictError):
            reg.register_result(factory_id=factory_id, key=key, result=res2)
