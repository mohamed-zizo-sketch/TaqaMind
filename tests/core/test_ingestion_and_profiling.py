"""Tests for CSV data ingestion and dataset profiling engines."""

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
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DataIngestion,
    DataProfiler,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    PipelineContext,
)
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.registry import LocalDatasetArtifactStorage


def make_contract_and_context() -> tuple[DatasetContract, PipelineContext]:
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
    )
    waste_qty = FeatureContract(
        name="waste_qty",
        display_name="Waste Quantity",
        description="Measured pre-production waste.",
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
        features=(batch_id, planned_qty),
        target=target,
        prediction_context=PredictionContext(
            description="Predict waste before production.",
        ),
        business_keys=("batch_id",),
    )
    dataset_revision_id = uuid4()
    context = PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=dataset_revision_id,
        domain="textile",
        process="weaving",
        dataset_id="weaving_batches",
        contract_version="1.0.0",
        task_id="waste_regression",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=DatasetPurpose.TRAINING,
    )
    return contract, context


def store_raw_csv(
    storage: LocalDatasetArtifactStorage,
    context: PipelineContext,
    raw_bytes: bytes,
) -> DatasetArtifact:
    ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(raw_bytes).hexdigest(),
        size_bytes=len(raw_bytes),
    )
    storage.put(factory_id=context.factory_id, reference=ref, stream=BytesIO(raw_bytes))
    metadata = DatasetMetadata.create_raw(
        context=context,
        source_file_sha256=ref.content_sha256,
    )
    return DatasetArtifact(metadata=metadata, reference=ref)


def test_engines_satisfy_protocols(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    assert isinstance(CsvDataIngestion(storage), DataIngestion)
    assert isinstance(PandasDataProfiler(storage), DataProfiler)


def test_ingest_and_profile_valid_csv(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    csv_bytes = b"batch_id,planned_qty,waste_qty\nB1,100.0,5.0\nB1,100.0,5.0\nB2,200.0,\n"
    raw_artifact = store_raw_csv(storage, context, csv_bytes)

    ingestion = CsvDataIngestion(storage)
    ingest_result = ingestion.ingest(
        context=context,
        artifact=raw_artifact,
        contract=contract,
    )

    assert ingest_result.status == ExecutionStatus.SUCCEEDED
    assert ingest_result.output_artifact is not None
    ingested = ingest_result.output_artifact
    assert ingested.metadata.state == DatasetState.INGESTED
    assert ingested.metadata.row_count == 3
    assert [c.name for c in ingested.metadata.columns] == [
        "batch_id",
        "planned_qty",
        "waste_qty",
    ]

    profiler = PandasDataProfiler(storage)
    prof_result = profiler.profile(
        context=context,
        artifact=ingested,
        contract=contract,
    )

    assert prof_result.result.status == ExecutionStatus.SUCCEEDED
    assert prof_result.profile is not None
    assert prof_result.profile.row_count == 3
    assert prof_result.profile.exact_duplicate_row_count == 1

    waste_col = next(c for c in prof_result.profile.columns if c.name == "waste_qty")
    assert waste_col.missing_count == 1
    assert waste_col.unique_count == 1
    assert waste_col.numeric_minimum == 5.0
    assert waste_col.numeric_maximum == 5.0


def test_ingest_rejects_duplicate_or_blank_headers(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    ingestion = CsvDataIngestion(storage)

    dup_artifact = store_raw_csv(storage, context, b"planned_qty,planned_qty\n10,20\n")
    dup_res = ingestion.ingest(context=context, artifact=dup_artifact, contract=contract)
    assert dup_res.status == ExecutionStatus.FAILED
    assert dup_res.error is not None
    assert dup_res.error.code == "CSV_DUPLICATE_HEADER"

    blank_artifact = store_raw_csv(storage, context, b"planned_qty,\n10,20\n")
    blank_res = ingestion.ingest(context=context, artifact=blank_artifact, contract=contract)
    assert blank_res.status == ExecutionStatus.FAILED
    assert blank_res.error is not None
    assert blank_res.error.code == "CSV_BLANK_HEADER"