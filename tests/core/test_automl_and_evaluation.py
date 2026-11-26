"""Tests for Core P1.7 AutoML data access, training, and protected evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from taqamind.core.automl import (
    ControlledAutoMLTrainer,
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
)
from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
    PredictionContext,
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.evaluation import ProtectedFinalEvaluator
from taqamind.core.feature_engineering import DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces.artifacts import (
    ArtifactFormat,
    ArtifactReference,
    calculate_bytes_sha256,
)
from taqamind.core.interfaces.context import PipelineContext
from taqamind.core.interfaces.data_access import DataAccessDeniedError
from taqamind.core.interfaces.dataset import DatasetArtifact, DatasetMetadata, DatasetState
from taqamind.core.interfaces.evaluation import (
    AcceptancePolicy,
    EvaluationDecision,
    FrozenModelSelection,
)
from taqamind.core.interfaces.lineage import RecordScope
from taqamind.core.interfaces.processing_models import (
    ApprovedMapping,
    ColumnMappingEntry,
    MappingMethod,
)
from taqamind.core.interfaces.splitting import SplitConfiguration, SplitStrategy
from taqamind.core.interfaces.training import (
    CandidateConfiguration,
    CandidateRole,
    ModelFamily,
    TrainingConfiguration,
)
from taqamind.core.interfaces.training_registry import (
    TrainingIntent,
    TrainingMode,
    TrainingReuseKey,
    calculate_split_fingerprint,
    calculate_training_configuration_fingerprint,
)
from taqamind.core.lineage import LocalRecordLineageRegistry, RecordMembershipBuilder
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.readiness import CoreTrainingReadinessChecker
from taqamind.core.registry import (
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalModelArtifactStorage,
)
from taqamind.core.splitting import (
    CoreDatasetSplitter,
    CoreLeakageValidator,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
)


def _build_contract() -> DatasetContract:
    run_id = FeatureContract(
        name="run_id",
        display_name="Run ID",
        description="Run identifier",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    lot_id = FeatureContract(
        name="lot_id",
        display_name="Lot ID",
        description="Lot group",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    run_ts = FeatureContract(
        name="run_ts",
        display_name="Run Timestamp",
        description="Timestamp",
        data_type=DataType.DATETIME,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    mass = FeatureContract(
        name="planned_mass_kg",
        display_name="Planned Mass",
        description="Planned mass in kg",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    line = FeatureContract(
        name="line_code",
        display_name="Line Code",
        description="Production line",
        data_type=DataType.STRING,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    scrap = FeatureContract(
        name="scrap_kg",
        display_name="Scrap Kg",
        description="Pre-production scrap quantity",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    return DatasetContract(
        dataset_id="batch_runs",
        version="1.0.0",
        domain="generic_domain",
        process="batch_process",
        description="P1.7 Core AutoML contract",
        granularity="batch_run",
        features=(run_id, lot_id, run_ts, mass, line),
        target=TargetContract(
            target_id="loss_prediction",
            feature=scrap,
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(definition="Scrap kg", granularity="batch_run"),
            confirmation=TargetConfirmation(
                status=ConfirmationStatus.CONFIRMED,
                confirmed_by="plant_qa",
                confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            derived_rate=RateDefinition(
                name="scrap_rate",
                numerator="scrap_kg",
                denominator="planned_mass_kg",
                granularity="batch_run",
                output_unit="%",
                multiplier=100.0,
            ),
        ),
        prediction_context=PredictionContext(description="Batch run prediction"),
        business_keys=("run_id",),
        grouping_columns=("lot_id",),
        time_column="run_ts",
    )


def _setup_split_and_registries(tmp_path: Path, n_rows: int = 36):
    contract = _build_contract()
    factory_id = uuid4()
    t0 = datetime.now(UTC) - timedelta(hours=1)
    rev_id = uuid4()
    ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=rev_id,
        domain="generic_domain",
        process="batch_process",
        task_id="loss_prediction",
        dataset_id="batch_runs",
        contract_version="1.0.0",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
        created_at=t0 + timedelta(minutes=10),
    )

    lines = ["run_id,lot_id,run_ts,planned_mass_kg,line_code,scrap_kg"]
    base_ts = datetime(2026, 1, 1, tzinfo=UTC)
    for idx in range(n_rows):
        lot = f"LOT_{idx // 2:02d}"
        ts = (base_ts + timedelta(hours=idx)).isoformat()
        mass = 100.0 + (idx % 7) * 50.0
        line = "L1" if idx % 2 == 0 else "L2"
        scrap = round(0.05 * mass + (12.0 if line == "L2" else 2.0), 3)
        lines.append(f"R{idx:03d},{lot},{ts},{mass},{line},{scrap}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    ds_storage = LocalDatasetArtifactStorage(tmp_path / "ds_storage")
    model_storage = LocalModelArtifactStorage(tmp_path / "model_storage")
    ds_registry = LocalDatasetRegistry(tmp_path / "ds_registry", ds_storage)
    lineage_registry = LocalRecordLineageRegistry(
        tmp_path / "lineage_registry",
        dataset_registry=ds_registry,
    )
    split_registry = LocalSplitRegistry(tmp_path / "split_registry")
    usage_registry = LocalRecordUsageRegistry(tmp_path / "usage_registry")
    training_registry = LocalTrainingRegistry(tmp_path / "training_registry")
    eval_registry = LocalEvaluationRegistry(
        root_dir=tmp_path / "eval_registry",
        split_registry=split_registry,
        usage_registry=usage_registry,
        lineage_registry=lineage_registry,
    )

    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=calculate_bytes_sha256(payload),
        size_bytes=len(payload),
    )
    ds_storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(payload))
    raw = DatasetArtifact(
        metadata=DatasetMetadata(
            revision_id=rev_id,
            root_revision_id=rev_id,
            factory_id=factory_id,
            domain=ctx.domain,
            process=ctx.process,
            dataset_id=ctx.dataset_id,
            contract_version=ctx.contract_version,
            task_id=ctx.task_id,
            task_version=ctx.task_version,
            target_version=ctx.target_version,
            source_file_sha256=ref.content_sha256,
            state=DatasetState.RAW,
        ),
        reference=ref,
    )
    ds_registry.register(factory_id=factory_id, artifact=raw)
    ing_res = CsvDataIngestion(ds_storage).ingest(context=ctx, artifact=raw, contract=contract)
    ing = ing_res.output_artifact
    assert ing is not None
    ds_registry.register(factory_id=factory_id, artifact=ing)

    mapper = DeterministicColumnMapper(ds_storage)
    prop = mapper.propose_mapping(context=ctx, artifact=ing, contract=contract)
    entries = tuple(
        ColumnMappingEntry(
            source_column=e.source_column,
            canonical_column=e.canonical_column,
            method=MappingMethod.EXACT,
            source_unit=contract.get_feature(e.canonical_column).unit,
            canonical_unit=contract.get_feature(e.canonical_column).unit,
            requires_confirmation=False,
            reason="Exact match.",
        )
        for e in prop.entries
    )
    mapped = mapper.apply_mapping(
        context=ctx,
        artifact=ing,
        contract=contract,
        mapping=ApprovedMapping(
            proposal=prop.model_copy(update={"entries": entries}),
            approved_by="test",
            confirmed_source_columns=tuple(e.source_column for e in entries),
        ),
    ).output_artifact
    assert mapped is not None
    ds_registry.register(factory_id=factory_id, artifact=mapped)

    cleaned = ContractDataCleaner(ds_storage).clean(
        context=ctx,
        artifact=mapped,
        contract=contract,
    ).output_artifact
    assert cleaned is not None
    ds_registry.register(factory_id=factory_id, artifact=cleaned)

    engineered = DeterministicFeatureEngineer(ds_storage).engineer(
        context=ctx,
        artifact=cleaned,
        contract=contract,
    ).output_artifact
    assert engineered is not None
    ds_registry.register(factory_id=factory_id, artifact=engineered)

    scope = RecordScope(
        factory_id=factory_id,
        domain=ctx.domain,
        process=ctx.process,
        dataset_id=ctx.dataset_id,
        identity_namespace="production_record",
    )
    builder = RecordMembershipBuilder(ds_storage, lineage_registry)
    builder.build_and_register(
        scope=scope,
        artifact=engineered,
        contract=contract,
    )
    membership_manifest = lineage_registry.get_membership(
        scope=scope,
        dataset_revision_id=engineered.metadata.revision_id,
    )

    readiness = CoreTrainingReadinessChecker(ds_storage, lineage_registry).assess(
        context=ctx,
        artifact=engineered,
        contract=contract,
    )
    readiness.require_split_eligibility()

    split_cfg = SplitConfiguration(
        strategy=SplitStrategy.GROUP_TEMPORAL,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        justification="Chronological lot-grouped split for P1.7 verification.",
    )
    splitter = CoreDatasetSplitter(ds_storage, lineage_registry)
    manifest = splitter.split(readiness=readiness, configuration=split_cfg)
    split_registry.register_split(factory_id=factory_id, manifest=manifest)

    leakage = CoreLeakageValidator(ds_storage, lineage_registry, usage_registry).validate(
        readiness=readiness,
        manifest=manifest,
    )
    leakage.require_training_eligibility()
    split_registry.register_leakage_validation(factory_id=factory_id, validation=leakage)
    auth = split_registry.issue_training_authorization(
        factory_id=factory_id,
        validation_id=leakage.validation_id,
    )
    return (
        contract,
        ctx,
        engineered,
        membership_manifest,
        manifest,
        auth,
        ds_storage,
        model_storage,
        ds_registry,
        lineage_registry,
        split_registry,
        usage_registry,
        training_registry,
        eval_registry,
    )


def test_p17_automl_all_families_and_protected_single_use_evaluation(tmp_path: Path) -> None:
    (
        contract,
        ctx,
        ing,
        membership_manifest,
        manifest,
        auth,
        ds_storage,
        model_storage,
        ds_registry,
        lineage_registry,
        split_registry,
        usage_registry,
        training_registry,
        eval_registry,
    ) = _setup_split_and_registries(tmp_path)

    dev_provider = LocalDevelopmentDataProvider(
        partition_root=tmp_path / "dev_partitions",
        storage=ds_storage,
        dataset_registry=ds_registry,
        lineage_registry=lineage_registry,
        split_registry=split_registry,
        usage_registry=usage_registry,
    )
    test_provider = LocalProtectedTestDataProvider(
        partition_root=tmp_path / "test_partitions",
        storage=ds_storage,
        dataset_registry=ds_registry,
        lineage_registry=lineage_registry,
        split_registry=split_registry,
        evaluation_registry=eval_registry,
    )

    policy = AcceptancePolicy(
        minimum_improvement_pct=1.0,
        maximum_test_mae=25.0,
        justification="Require >= 1% MAE improvement over baseline.",
        created_at=ctx.created_at - timedelta(minutes=5),
    )
    config = TrainingConfiguration(
        candidates=(
            CandidateConfiguration(family=ModelFamily.DUMMY_REGRESSOR, role=CandidateRole.BASELINE),
            CandidateConfiguration(family=ModelFamily.HISTORICAL_RATE, role=CandidateRole.BASELINE),
            CandidateConfiguration(family=ModelFamily.RIDGE, role=CandidateRole.CHALLENGER),
            CandidateConfiguration(family=ModelFamily.ELASTIC_NET, role=CandidateRole.CHALLENGER),
            CandidateConfiguration(family=ModelFamily.RANDOM_FOREST, role=CandidateRole.CHALLENGER),
            CandidateConfiguration(
                family=ModelFamily.HIST_GRADIENT_BOOSTING,
                role=CandidateRole.CHALLENGER,
            ),
            CandidateConfiguration(
                family=ModelFamily.HURDLE_REGRESSION,
                role=CandidateRole.CHALLENGER,
            ),
            CandidateConfiguration(family=ModelFamily.XGBOOST, role=CandidateRole.CHALLENGER),
        ),
        justification="Verify all 8 Core model families.",
    )

    intent = TrainingIntent(
        context=ctx,
        mode=TrainingMode.INITIAL,
        requested_by="core-test",
        reason="Initial P1.7 Core training verification.",
    )
    training_registry.register_intent(factory_id=ctx.factory_id, intent=intent)

    trainer = ControlledAutoMLTrainer(
        development_data_provider=dev_provider,
        model_storage=model_storage,
    )
    run_result = trainer.train(
        authorization=auth,
        contract=contract,
        configuration=config,
        policy=policy,
    )

    assert len(run_result.results) == 8
    assert run_result.eligible_for_final_evaluation
    assert run_result.best_challenger is not None
    assert run_result.best_baseline is not None

    reuse_key = TrainingReuseKey.from_intent(
        intent=intent,
        dataset_content_sha256=ing.reference.content_sha256,
        contract_sha256=calculate_bytes_sha256(contract.model_dump_json().encode("utf-8")),
        split_sha256=calculate_split_fingerprint(manifest, membership_manifest),
        training_configuration_sha256=calculate_training_configuration_fingerprint(config, policy),
        implementation_sha256=calculate_bytes_sha256(b"core-automl-v1"),
    )
    training_registry.register_result(factory_id=ctx.factory_id, key=reuse_key, result=run_result)
    reused = training_registry.find_reusable_result(key=reuse_key)
    assert reused is not None
    assert reused.training_run_id == run_result.training_run_id

    selection = FrozenModelSelection.from_training_run(training_run=run_result, policy=policy)
    evaluator = ProtectedFinalEvaluator(
        contract=contract,
        evaluation_registry=eval_registry,
        test_data_provider=test_provider,
        model_storage=model_storage,
    )
    eval_result = evaluator.evaluate(selection=selection, manifest=manifest)
    assert eval_result.decision == EvaluationDecision.ACCEPTED
    assert eval_result.selected_mae < eval_result.baseline_mae

    with pytest.raises(DataAccessDeniedError):
        evaluator.evaluate(selection=selection, manifest=manifest)
