"""Core P1.6 tests for readiness, splitting, leakage validation, and split/usage registries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

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
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.feature_engineering import DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    LeakageCheck,
    LeakageCheckStatus,
    PipelineContext,
    RecordScope,
    RecordUsage,
    RecordUsageRole,
    SplitConfiguration,
    SplitStrategy,
)
from taqamind.core.lineage import LocalRecordLineageRegistry, RecordMembershipBuilder
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.readiness import CoreTrainingReadinessChecker
from taqamind.core.registry import LocalDatasetArtifactStorage, LocalDatasetRegistry
from taqamind.core.splitting import (
    CoreDatasetSplitter,
    CoreLeakageValidator,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
)


def _prepare_engineered(
    tmp_path: Path,
    *,
    confirmed: bool = True,
    n_rows: int = 18,
) -> tuple[
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalRecordLineageRegistry,
    DatasetContract,
    PipelineContext,
    DatasetArtifact,
]:
    storage = LocalDatasetArtifactStorage(tmp_path / "artifacts")
    ds_reg = LocalDatasetRegistry(tmp_path / "reg", storage)
    lin_reg = LocalRecordLineageRegistry(tmp_path / "reg", dataset_registry=ds_reg)

    factory_id = uuid4()
    rec_id = FeatureContract(
        name="record_key",
        display_name="Record Key",
        description="Key",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    order_grp = FeatureContract(
        name="order_group",
        display_name="Order Group",
        description="Group",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    event_ts = FeatureContract(
        name="event_date",
        display_name="Event Date",
        description="Date",
        data_type=DataType.DATE,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    qty = FeatureContract(
        name="planned_qty",
        display_name="Planned Qty",
        description="Planned",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
    )
    waste = FeatureContract(
        name="waste_qty",
        display_name="Waste Qty",
        description="Target",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    conf = (
        TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="plant_qa",
            confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        if confirmed
        else TargetConfirmation()
    )
    contract = DatasetContract(
        dataset_id="ds_p16",
        version="1.0.0",
        domain="industrial",
        process="batch",
        description="P1.6 test",
        granularity="batch",
        features=(rec_id, order_grp, event_ts, qty),
        target=TargetContract(
            target_id="t_p16",
            feature=waste,
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(definition="Waste", granularity="batch"),
            confirmation=conf,
        ),
        prediction_context=PredictionContext(description="Pred"),
        business_keys=("record_key",),
        grouping_columns=("order_group",),
        time_column="event_date",
    )

    lines = ["record_key,order_group,event_date,planned_qty,waste_qty"]
    base_dt = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(n_rows):
        grp = f"GRP-{i // 2:03d}"
        dt_str = (base_dt + timedelta(days=i)).date().isoformat()
        w_val = 2.0 + (i % 4) if i % 3 == 0 else 0.0
        lines.append(f"R-{i:04d},{grp},{dt_str},{100.0 + i * 5},{w_val}")
    raw_bytes = ("\n".join(lines) + "\n").encode("utf-8")

    rev_id = uuid4()
    ctx = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=rev_id,
        domain="industrial",
        process="batch",
        task_id="t_p16",
        dataset_id="ds_p16",
        contract_version="1.0.0",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
    )
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(raw_bytes).hexdigest(),
        size_bytes=len(raw_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(raw_bytes))
    raw = DatasetArtifact(
        metadata=DatasetMetadata(
            revision_id=rev_id,
            root_revision_id=rev_id,
            factory_id=factory_id,
            domain="industrial",
            process="batch",
            dataset_id="ds_p16",
            contract_version="1.0.0",
            task_id="t_p16",
            task_version="1.0.0",
            target_version="1.0.0",
            source_file_sha256=ref.content_sha256,
            state=DatasetState.RAW,
        ),
        reference=ref,
    )
    ds_reg.register(factory_id=factory_id, artifact=raw)
    ing_res = CsvDataIngestion(storage).ingest(context=ctx, artifact=raw, contract=contract)
    ing = ing_res.output_artifact
    assert ing is not None
    ds_reg.register(factory_id=factory_id, artifact=ing)

    mapper = DeterministicColumnMapper(storage)
    prop = mapper.propose_mapping(context=ctx, artifact=ing, contract=contract)
    entries = tuple(
        e.model_copy(update={"source_unit": e.canonical_unit})
        if e.canonical_unit is not None and e.source_unit is None
        else e
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
    ds_reg.register(factory_id=factory_id, artifact=mapped)

    cleaned = ContractDataCleaner(storage).clean(
        context=ctx, artifact=mapped, contract=contract
    ).output_artifact
    assert cleaned is not None
    ds_reg.register(factory_id=factory_id, artifact=cleaned)

    engineered = DeterministicFeatureEngineer(storage).engineer(
        context=ctx, artifact=cleaned, contract=contract
    ).output_artifact
    assert engineered is not None
    ds_reg.register(factory_id=factory_id, artifact=engineered)

    scope = RecordScope(
        factory_id=factory_id,
        domain="industrial",
        process="batch",
        dataset_id="ds_p16",
        identity_namespace="production_record",
    )
    RecordMembershipBuilder(storage, lin_reg).build_and_register(
        scope=scope, artifact=engineered, contract=contract
    )
    return storage, ds_reg, lin_reg, contract, ctx, engineered


def test_p16_readiness_blocks_unconfirmed_target(tmp_path: Path) -> None:
    storage, _, lin_reg, contract, ctx, engineered = _prepare_engineered(
        tmp_path, confirmed=False
    )
    checker = CoreTrainingReadinessChecker(storage, lin_reg)
    readiness = checker.assess(context=ctx, artifact=engineered, contract=contract)
    with pytest.raises(ValueError, match="not eligible"):
        readiness.require_split_eligibility()


def test_p16_group_temporal_split_leakage_and_historical_test_exposure(tmp_path: Path) -> None:
    storage, _, lin_reg, contract, ctx, engineered = _prepare_engineered(
        tmp_path, confirmed=True, n_rows=18
    )
    split_reg = LocalSplitRegistry(tmp_path / "reg")
    usage_reg = LocalRecordUsageRegistry(tmp_path / "reg")

    readiness = CoreTrainingReadinessChecker(storage, lin_reg).assess(
        context=ctx, artifact=engineered, contract=contract
    )
    readiness.require_split_eligibility()

    splitter = CoreDatasetSplitter(storage, lin_reg)
    manifest = splitter.split(
        readiness=readiness,
        configuration=SplitConfiguration(
            strategy=SplitStrategy.GROUP_TEMPORAL,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            justification="Group-temporal split test.",
        ),
    )
    split_reg.register_split(factory_id=ctx.factory_id, manifest=manifest)

    validator = CoreLeakageValidator(storage, lin_reg, usage_reg)
    leakage = validator.validate(readiness=readiness, manifest=manifest)
    leakage.require_training_eligibility()
    split_reg.register_leakage_validation(factory_id=ctx.factory_id, validation=leakage)
    auth = split_reg.issue_training_authorization(
        factory_id=ctx.factory_id, validation_id=leakage.validation_id
    )
    split_reg.verify_training_authorization(factory_id=ctx.factory_id, authorization=auth)

    # Mark one test record as historically exposed -> subsequent validation must BLOCK it.
    scope = RecordScope(
        factory_id=ctx.factory_id,
        domain="industrial",
        process="batch",
        dataset_id="ds_p16",
        identity_namespace="production_record",
    )
    mem = lin_reg.get_membership(scope=scope, dataset_revision_id=engineered.metadata.revision_id)
    test_rid = manifest.test.record_ids[0]
    ver_id = next(m.record_version_id for m in mem.memberships if m.record_id == test_rid)
    usage_reg.record_usage(
        scope=scope,
        usages=(
            RecordUsage(
                scope=scope,
                record_id=test_rid,
                record_version_id=ver_id,
                execution_id=uuid4(),
                split_id=manifest.split_id,
                role=RecordUsageRole.PROTECTED_TEST,
            ),
        ),
    )

    leakage2 = validator.validate(readiness=readiness, manifest=manifest)
    with pytest.raises(ValueError, match="does not permit"):
        leakage2.require_training_eligibility()
    by_check = {c.check: c.status for c in leakage2.checks}
    assert by_check[LeakageCheck.HISTORICAL_USAGE] == LeakageCheckStatus.FAILED
