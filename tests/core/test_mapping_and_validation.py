"""Tests for column mapping and semantic validation engines."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

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
from taqamind.core.ingestion.safe_csv import SafeCsvReader
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
from taqamind.core.interfaces.processing_models import MappingProposal
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.core.validation import ContractSemanticValidator


def approve_mapping_with_units(
    proposal: MappingProposal,
    *,
    source_units: dict[str, str],
    approved_by: str = "plant_engineer",
) -> ApprovedMapping:
    """Simulate explicit confirmation of source units by a factory operator."""
    entries = tuple(
        entry.model_copy(update={"source_unit": source_units.get(entry.source_column)})
        for entry in proposal.entries
    )
    return ApprovedMapping(
        proposal=proposal.model_copy(update={"entries": entries}),
        approved_by=approved_by,
        confirmed_source_columns=tuple(
            entry.source_column
            for entry in entries
            if entry.requires_confirmation or entry.canonical_unit is not None
        ),
    )


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

    approved = approve_mapping_with_units(
        proposal, source_units={"Planned Qty": "kg", "waste_qty": "kg"}
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
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": "kg", "waste_qty": "kg"}
    )
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
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": "kg"}, approved_by="operator"
    )
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


@pytest.mark.parametrize(
    ("source_unit", "expected"),
    [("g", 5.0), ("kg", 5000.0)],
)
def test_mapping_converts_confirmed_units(tmp_path: Path, source_unit: str, expected: float):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    ingested = ingest_csv(
        storage, context, contract, b"batch_id,planned_qty,waste_qty\nB1,5000,2000\nB2,0,0\n"
    )
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    by_source = {entry.source_column: entry for entry in proposal.entries}
    assert by_source["planned_qty"].source_unit is None
    assert by_source["planned_qty"].requires_confirmation
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": source_unit, "waste_qty": "g"}
    )
    result = mapper.apply_mapping(
        context=context, artifact=ingested, contract=contract, mapping=approved
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_artifact is not None
    assert result.report is not None
    with storage.open_verified(
        factory_id=context.factory_id, reference=result.output_artifact.reference
    ) as stream:
        mapped = pd.read_csv(stream)
    assert mapped["planned_qty"].tolist() == pytest.approx([expected, 0.0])
    assert mapped["waste_qty"].tolist() == pytest.approx([2.0, 0.0])
    assert mapped["batch_id"].tolist() == ["B1", "B2"]
    assert "UNIT_CONVERSION_APPLIED" in {issue.code for issue in result.report.issues}


@pytest.mark.parametrize("source_unit", ["L", "ton"])
def test_mapping_rejects_unsupported_conversion(tmp_path: Path, source_unit: str):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    ingested = ingest_csv(storage, context, contract, b"batch_id,planned_qty,waste_qty\nB1,5,1\n")
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": source_unit, "waste_qty": "kg"}
    )
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    result = mapper.apply_mapping(
        context=context, artifact=ingested, contract=contract, mapping=approved
    )
    assert result.status == ExecutionStatus.FAILED
    assert result.output_artifact is None
    assert result.error is not None
    assert result.error.code == "UNIT_CONVERSION_FAILED"
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == before


def test_mapping_requires_explicit_unit_confirmation(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    ingested = ingest_csv(storage, context, contract, b"batch_id,planned_qty,waste_qty\nB1,5,1\n")
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    with pytest.raises(ValueError, match="Required mapping confirmations are missing"):
        ApprovedMapping(proposal=proposal, approved_by="operator")
    approved = approve_mapping_with_units(proposal, source_units={"waste_qty": "kg"})
    with pytest.raises(ValueError, match="Unknown source unit"):
        mapper.apply_mapping(
            context=context, artifact=ingested, contract=contract, mapping=approved
        )


@pytest.mark.parametrize(
    ("canonical_unit", "source_unit", "data_type", "value", "expected"),
    [
        ("kg", "g", DataType.INTEGER, "2000", 2),
        ("kg", "g", DataType.INTEGER, "1500", None),
        ("kg", "g", DataType.INTEGER, "9007199254740993", None),
        ("rpm", "rpm", DataType.FLOAT, "300", 300),
        ("kg/h", "kg/h", DataType.FLOAT, "5", 5),
        ("g/h", "kg/h", DataType.FLOAT, "5", None),
    ],
)
def test_mapping_integer_and_compound_units(
    tmp_path, canonical_unit, source_unit, data_type, value, expected
):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    payload = contract.model_dump()
    payload["features"][1]["unit"] = canonical_unit
    payload["features"][1]["data_type"] = data_type
    contract = DatasetContract.model_validate(payload)
    csv_bytes = f"batch_id,planned_qty,waste_qty\nB1,{value},1\n".encode()
    ingested = ingest_csv(storage, context, contract, csv_bytes)
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": source_unit, "waste_qty": "kg"}
    )
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    result = mapper.apply_mapping(
        context=context, artifact=ingested, contract=contract, mapping=approved
    )
    if expected is None:
        assert result.status == ExecutionStatus.FAILED
        assert result.output_artifact is None
        assert result.error is not None
        assert result.error.code == "UNIT_CONVERSION_FAILED"
        assert {path for path in tmp_path.rglob("*") if path.is_file()} == before
    else:
        assert result.status == ExecutionStatus.SUCCEEDED
        assert result.output_artifact is not None
        with storage.open_verified(
            factory_id=context.factory_id, reference=result.output_artifact.reference
        ) as stream:
            assert pd.read_csv(stream)["planned_qty"].iloc[0] == expected


@pytest.mark.parametrize("header", ["batch_id", "Batch Number"])
def test_mapping_preserves_lexical_values(tmp_path: Path, header: str):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    csv_bytes = (
        f"{header},planned_qty,waste_qty\n"
        "000123,9007199254740993,1.2300\n"
        "00123,2.000,0\n"
        "NA,3,1\n"
        "NULL,4,\n"
    ).encode()
    ingested = ingest_csv(storage, context, contract, csv_bytes)
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": "kg", "waste_qty": "kg"}
    )
    result = mapper.apply_mapping(
        context=context, artifact=ingested, contract=contract, mapping=approved
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_artifact is not None
    with storage.open_verified(
        factory_id=context.factory_id, reference=result.output_artifact.reference
    ) as stream:
        mapped = SafeCsvReader().read(stream.read())
    assert mapped["batch_id"].tolist() == ["000123", "00123", "NA", "NULL"]
    assert mapped["planned_qty"].tolist() == ["9007199254740993", "2.000", "3", "4"]
    assert mapped["waste_qty"].iloc[:3].tolist() == ["1.2300", "0", "1"]
    assert pd.isna(mapped["waste_qty"].iloc[3])


def test_mapping_rejects_large_fraction_without_ingestion_rounding(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    payload = contract.model_dump()
    payload["features"][1]["data_type"] = DataType.INTEGER
    contract = DatasetContract.model_validate(payload)
    ingested = ingest_csv(
        storage, context, contract,
        b"batch_id,planned_qty,waste_qty\nB1,9007199254740001,1\n",
    )
    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = approve_mapping_with_units(
        proposal, source_units={"planned_qty": "g", "waste_qty": "kg"}
    )
    result = mapper.apply_mapping(
        context=context, artifact=ingested, contract=contract, mapping=approved
    )
    assert result.status == ExecutionStatus.FAILED
    assert result.output_artifact is None
    assert result.error is not None
    assert result.error.code == "UNIT_CONVERSION_FAILED"
