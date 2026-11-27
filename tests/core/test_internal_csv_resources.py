"""Regression tests for bounded internal CSV reads."""

from hashlib import sha256
from io import BytesIO
from uuid import uuid4

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
from taqamind.core.feature_engineering.feature_engineer import DeterministicFeatureEngineer
from taqamind.core.ingestion.artifact_csv import csv_configuration_sha256, read_verified_csv
from taqamind.core.ingestion.resource_policy import (
    DEFAULT_INTERNAL_CSV_POLICY,
    CsvResourcePolicy,
    resolve_internal_csv_policy,
)
from taqamind.core.ingestion.safe_csv import SafeCsvError, SafeCsvReader
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetMetadata,
    PipelineContext,
)
from taqamind.core.interfaces.storage import ArtifactScopeError, ArtifactSizeLimitError
from taqamind.core.mapping.column_mapper import DeterministicColumnMapper
from taqamind.core.preprocessing.data_cleaner import ContractDataCleaner
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.registry import LocalDatasetArtifactStorage
from taqamind.core.validation.semantic_validator import ContractSemanticValidator


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


def test_internal_defaults_and_legacy_limit():
    assert DEFAULT_INTERNAL_CSV_POLICY == CsvResourcePolicy(
        max_bytes=104_857_600, max_rows=250_000, max_columns=1000, max_cells=10_000_000
    )
    assert resolve_internal_csv_policy(max_rows=3).max_rows == 3
    with pytest.raises(ValueError):
        resolve_internal_csv_policy(CsvResourcePolicy(), max_rows=3)


@pytest.mark.parametrize(
    "component",
    [
        PandasDataProfiler,
        DeterministicColumnMapper,
        ContractSemanticValidator,
        ContractDataCleaner,
        DeterministicFeatureEngineer,
    ],
)
def test_consumers_accept_policy(tmp_path, component):
    policy = CsvResourcePolicy(max_bytes=64)
    service = component(LocalDatasetArtifactStorage(tmp_path), resource_policy=policy)
    assert service._resource_policy == policy


def test_precheck_rejects_before_open(tmp_path, monkeypatch):
    storage = LocalDatasetArtifactStorage(tmp_path)
    _, context = make_contract_and_context()
    raw = make_raw(storage, context, b"id,value\n001,NA\n")
    policy = CsvResourcePolicy(max_bytes=1)

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized artifact opened")

    monkeypatch.setattr(storage, "open_verified", forbidden)
    with pytest.raises(SafeCsvError) as error:
        read_verified_csv(
            storage=storage,
            factory_id=context.factory_id,
            reference=raw.reference,
            reader=SafeCsvReader(resource_policy=policy),
            policy=policy,
        )
    assert error.value.code == "CSV_BYTE_LIMIT_EXCEEDED"
    with pytest.raises(ArtifactScopeError):
        read_verified_csv(
            storage=storage,
            factory_id=uuid4(),
            reference=raw.reference,
            reader=SafeCsvReader(resource_policy=policy),
            policy=policy,
        )


def test_verified_read_preserves_tokens(tmp_path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    _, context = make_contract_and_context()
    payload = b"id,value\n001,NA\n"
    raw = make_raw(storage, context, payload)
    policy = CsvResourcePolicy(max_bytes=len(payload))
    frame = read_verified_csv(
        storage=storage,
        factory_id=context.factory_id,
        reference=raw.reference,
        reader=SafeCsvReader(resource_policy=policy),
        policy=policy,
    )
    assert frame.iloc[0].tolist() == ["001", "NA"]


def test_storage_limit_is_structured(tmp_path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    _, context = make_contract_and_context()
    raw = make_raw(storage, context, b"id,value\n001,NA\n")
    restricted = LocalDatasetArtifactStorage(tmp_path, max_size_bytes=1)
    policy = DEFAULT_INTERNAL_CSV_POLICY
    with pytest.raises(SafeCsvError) as error:
        read_verified_csv(
            storage=restricted,
            factory_id=context.factory_id,
            reference=raw.reference,
            reader=SafeCsvReader(resource_policy=policy),
            policy=policy,
        )
    assert error.value.code == "ARTIFACT_SIZE_LIMIT_EXCEEDED"
    assert isinstance(error.value.__cause__, ArtifactSizeLimitError)


def test_configuration_fingerprint():
    policy = DEFAULT_INTERNAL_CSV_POLICY
    assert csv_configuration_sha256(policy) == csv_configuration_sha256(policy)
    assert csv_configuration_sha256(policy) != csv_configuration_sha256(CsvResourcePolicy())
    assert csv_configuration_sha256(policy, extra={"mapping": 1}) != csv_configuration_sha256(
        policy, extra={"mapping": 2}
    )
