"""Integration tests for lossless CSV ingestion and structural rejection."""

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
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.ingestion.safe_csv import SafeCsvReader
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    PipelineContext,
)
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.registry import LocalDatasetArtifactStorage


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



def make_raw(storage, context, payload):
    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=context.factory_id, reference=reference, stream=BytesIO(payload))
    return DatasetArtifact(
        metadata=DatasetMetadata.create_raw(
            context=context, source_file_sha256=reference.content_sha256
        ),
        reference=reference,
    )


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_ingestion_preserves_lexical_values(tmp_path: Path, delimiter: str):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    payload = (
        f"Batch Number{delimiter}Planned Qty{delimiter}shift_code{delimiter}waste_qty\n"
        f"000123{delimiter}9007199254740993{delimiter}NA{delimiter}1.2300\n"
        f"00123{delimiter}{delimiter}NULL{delimiter}0\n"
        f'000124{delimiter}2{delimiter}"a\rb"{delimiter}3\n'
    ).encode()
    raw = make_raw(storage, context, payload)
    result = CsvDataIngestion(storage, delimiter=delimiter).ingest(
        context=context, artifact=raw, contract=contract
    )
    assert result.status == ExecutionStatus.SUCCEEDED
    assert result.output_artifact is not None
    output = result.output_artifact
    assert output.metadata.state == DatasetState.INGESTED
    assert output.metadata.row_count == 3
    with storage.open_verified(factory_id=context.factory_id, reference=output.reference) as stream:
        restored = SafeCsvReader().read(stream.read())
    expected = SafeCsvReader(delimiter=delimiter).read(payload)
    pd.testing.assert_frame_equal(restored, expected)
    with storage.open_verified(factory_id=context.factory_id, reference=raw.reference) as stream:
        assert stream.read() == payload


@pytest.mark.parametrize(
    ("payload", "code", "max_rows"),
    [
        (b"id,value\n1,2,3\n", "CSV_ROW_WIDTH_MISMATCH", None),
        (b"id,value\n1\n", "CSV_ROW_WIDTH_MISMATCH", None),
        (b"id,id\n1,2\n", "CSV_DUPLICATE_HEADER", None),
        (b"id,value\n1,2\n2,3\n", "CSV_ROW_LIMIT_EXCEEDED", 1),
        (b'id,value\n1,"unfinished\n', "CSV_PARSE_ERROR", None),
        (b"id,value\n1,\xff\n", "CSV_DECODE_ERROR", None),
    ],
)
def test_ingestion_rejects_invalid_csv_without_output(tmp_path: Path, payload, code, max_rows):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    raw = make_raw(storage, context, payload)
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    result = CsvDataIngestion(storage, max_rows=max_rows).ingest(
        context=context, artifact=raw, contract=contract
    )
    assert result.status == ExecutionStatus.FAILED
    assert result.output_artifact is None
    assert result.error is not None
    assert result.error.code == code
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == before


def test_profile_preserves_lexical_identity_and_schema(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    raw = make_raw(
        storage, context,
        b"batch_id,planned_qty,shift_code,waste_qty\n"
        b"000123,100,NA,5\n00123,100,NULL,5\n000123,100,NA,5\nNA,200,,\n",
    )
    ingested = CsvDataIngestion(storage).ingest(
        context=context, artifact=raw, contract=contract
    ).output_artifact
    assert ingested is not None
    result = PandasDataProfiler(storage).profile(
        context=context, artifact=ingested, contract=contract
    )
    assert result.result.status == ExecutionStatus.SUCCEEDED
    assert result.profile is not None
    profile = result.profile
    columns = {col.name: col for col in profile.columns}
    assert profile.row_count == 4
    assert profile.exact_duplicate_row_count == 1
    assert columns["batch_id"].unique_count == 3
    assert columns["batch_id"].missing_count == 0
    assert columns["batch_id"].numeric_minimum is None
    assert columns["shift_code"].unique_count == 2
    assert columns["shift_code"].missing_count == 1
    assert columns["planned_qty"].numeric_minimum == 100.0
    assert columns["planned_qty"].numeric_maximum == 200.0
    assert columns["waste_qty"].missing_count == 1
    assert tuple((c.name, c.physical_dtype) for c in profile.columns) == tuple(
        (c.name, c.physical_dtype) for c in ingested.metadata.columns
    )
