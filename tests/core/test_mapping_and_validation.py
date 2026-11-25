"""Tests for column mapping and semantic validation engines."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from taqamind.core.contracts import (
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
    ReadinessStatus,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    ColumnMapper,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    MappingMethod,
    PipelineContext,
    SemanticValidator,
)
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.core.validation import ContractSemanticValidator


def make_contract_and_context(
    *,
    purpose: DatasetPurpose = DatasetPurpose.TRAINING,
) -> tuple[DatasetContract, PipelineContext]:
    factory_id = uuid4()
    batch_id = FeatureContract(
        name="batch_id",
        display_name="Batch ID",
        description="Unique production batch identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Batch Number",),
    )
    planned_qty = FeatureContract(
        name="planned_qty",
        display_name="Planned Quantity",
        description="Planned production quantity.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Planned Qty",),
    )
    shift_code = FeatureContract(
        name="shift_code",
        display_name="Shift Code",
        description="Production shift.",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(allowed_values=("A", "B", "C")),
    )
    waste_qty = FeatureContract(
        name="waste_qty",
        display_name="Waste Quantity",
        description="Measured production waste.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    target = TargetContract(
        target_id="waste_regression",
        feature=waste_qty,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition="Measured production waste per batch.",
            granularity="batch",
        ),
    )
    contract = DatasetContract(
        dataset_id="weaving_batches",
        version="1.0.0",
        domain="textile",
        process="weaving",
        description="Synthetic weaving batch dataset.",
        granularity="batch",
        features=(batch_id, planned_qty, shift_code),
        target=target,
        prediction_context=PredictionContext(
            description="Predict waste before production.",
        ),
        business_keys=("batch_id",),
    )
    context = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=uuid4(),
        domain="textile",
        process="weaving",
        dataset_id="weaving_batches",
        contract_version="1.0.0",
        task_id="waste_regression",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=purpose,
        model_version_id=(uuid4() if purpose == DatasetPurpose.PREDICTION else None),
    )
    return contract, context


def ingest_csv(
    storage: LocalDatasetArtifactStorage,
    context: PipelineContext,
    contract: DatasetContract,
    csv_bytes: bytes,
) -> DatasetArtifact:
    ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(csv_bytes).hexdigest(),
        size_bytes=len(csv_bytes),
    )
    storage.put(factory_id=context.factory_id, reference=ref, stream=BytesIO(csv_bytes))
    raw_artifact = DatasetArtifact(
        metadata=DatasetMetadata.create_raw(
            context=context,
            source_file_sha256=ref.content_sha256,
        ),
        reference=ref,
    )
    res = CsvDataIngestion(storage).ingest(
        context=context,
        artifact=raw_artifact,
        contract=contract,
    )
    assert res.status == ExecutionStatus.SUCCEEDED
    assert res.output_artifact is not None
    return res.output_artifact


def test_engines_satisfy_protocols(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    assert isinstance(DeterministicColumnMapper(storage), ColumnMapper)
    assert isinstance(ContractSemanticValidator(storage), SemanticValidator)


def test_propose_apply_mapping_and_semantic_validation_succeed(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    csv_bytes = (
        b"Batch Number,Planned Qty,shift_code,waste_qty,extra_notes\n"
        b"B1,100.0,A,5.0,ok\n"
        b"B2,200.0,,10.0,none\n"
    )
    ingested = ingest_csv(storage, context, contract, csv_bytes)

    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(
        context=context,
        artifact=ingested,
        contract=contract,
    )

    by_source = {entry.source_column: entry for entry in proposal.entries}
    assert by_source["Batch Number"].canonical_column == "batch_id"
    assert by_source["Batch Number"].method == MappingMethod.ALIAS
    assert by_source["Batch Number"].requires_confirmation is True
    assert by_source["Planned Qty"].canonical_column == "planned_qty"
    assert by_source["shift_code"].method == MappingMethod.EXACT
    assert by_source["shift_code"].requires_confirmation is False
    assert by_source["waste_qty"].method == MappingMethod.EXACT
    assert proposal.unmapped_columns == ("extra_notes",)

    approved = ApprovedMapping(
        proposal=proposal,
        approved_by="plant_engineer",
        confirmed_source_columns=("Batch Number", "Planned Qty"),
    )
    map_res = mapper.apply_mapping(
        context=context,
        artifact=ingested,
        contract=contract,
        mapping=approved,
    )
    assert map_res.status == ExecutionStatus.SUCCEEDED
    assert map_res.output_artifact is not None
    mapped = map_res.output_artifact
    assert mapped.metadata.state == DatasetState.MAPPED
    assert [col.name for col in mapped.metadata.columns] == [
        "batch_id",
        "planned_qty",
        "shift_code",
        "waste_qty",
    ]

    validator = ContractSemanticValidator(storage)
    val_res = validator.validate(
        context=context,
        artifact=mapped,
        contract=contract,
    )
    assert val_res.status == ExecutionStatus.SUCCEEDED
    assert val_res.output_artifact is None
    assert val_res.report is not None
    assert val_res.report.status == ReadinessStatus.READY
    codes = [issue.code for issue in val_res.report.issues]
    assert "MISSING_VALUES_PENDING_IMPUTATION" in codes


def test_semantic_validation_blocks_constraints_missing_and_duplicates(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    csv_bytes = (
        b"batch_id,planned_qty,shift_code,waste_qty\n"
        b"B1,0.0,A,5.0\n"
        b"B1,150.0,Z,\n"
    )
    ingested = ingest_csv(storage, context, contract, csv_bytes)
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = ApprovedMapping(proposal=proposal, approved_by="plant_engineer")
    mapped = mapper.apply_mapping(
        context=context,
        artifact=ingested,
        contract=contract,
        mapping=approved,
    ).output_artifact
    assert mapped is not None

    val_res = ContractSemanticValidator(storage).validate(
        context=context,
        artifact=mapped,
        contract=contract,
    )
    assert val_res.status == ExecutionStatus.BLOCKED
    assert val_res.output_artifact is None
    assert val_res.report is not None
    assert val_res.report.status == ReadinessStatus.BLOCKED
    codes = {issue.code for issue in val_res.report.issues}
    assert {
        "DUPLICATE_BUSINESS_KEY",
        "CONSTRAINT_RANGE_VIOLATION",
        "CONSTRAINT_ALLOWED_VALUES_VIOLATION",
        "MISSING_VALUES_REJECTED",
    } <= codes


def test_prediction_mapping_excludes_target_column(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context(purpose=DatasetPurpose.PREDICTION)
    csv_bytes = b"batch_id,planned_qty,waste_qty\nB1,100.0,5.0\n"
    ingested = ingest_csv(storage, context, contract, csv_bytes)

    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    assert "waste_qty" in proposal.unmapped_columns
    approved = ApprovedMapping(proposal=proposal, approved_by="operator")
    map_res = mapper.apply_mapping(
        context=context,
        artifact=ingested,
        contract=contract,
        mapping=approved,
    )
    assert map_res.status == ExecutionStatus.SUCCEEDED
    assert map_res.output_artifact is not None
    assert [c.name for c in map_res.output_artifact.metadata.columns] == [
        "batch_id",
        "planned_qty",
    ]
