"""Tests for DataCleaner, FeatureEngineer, and train-only TabularPreprocessor."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
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
from taqamind.core.feature_engineering import (
    DerivedFeatureSpec,
    DeterministicFeatureEngineer,
)
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ApprovedMapping,
    ArtifactFormat,
    ArtifactReference,
    DataCleaner,
    DatasetArtifact,
    DatasetMetadata,
    DatasetState,
    ExecutionStatus,
    FeatureEngineer,
    PipelineContext,
)
from taqamind.core.mapping import DeterministicColumnMapper
from taqamind.core.preprocessing import ContractDataCleaner, TabularPreprocessor
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
    machine_speed = FeatureContract(
        name="machine_speed",
        display_name="Machine Speed",
        description="Configured machine speed.",
        data_type=DataType.FLOAT,
        unit="rpm",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
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
    speed_per_kg = FeatureContract(
        name="speed_per_kg",
        display_name="Speed Per Kg",
        description="Derived ratio of machine speed to planned quantity.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        derived_from=("machine_speed", "planned_qty"),
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
        features=(batch_id, planned_qty, machine_speed, shift_code, speed_per_kg),
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


def map_csv(
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
    raw = DatasetArtifact(
        metadata=DatasetMetadata.create_raw(
            context=context,
            source_file_sha256=ref.content_sha256,
        ),
        reference=ref,
    )
    ingested = CsvDataIngestion(storage).ingest(
        context=context,
        artifact=raw,
        contract=contract,
    ).output_artifact
    assert ingested is not None

    mapper = DeterministicColumnMapper(storage)
    proposal = mapper.propose_mapping(context=context, artifact=ingested, contract=contract)
    approved = ApprovedMapping(proposal=proposal, approved_by="engineer")
    mapped = mapper.apply_mapping(
        context=context,
        artifact=ingested,
        contract=contract,
        mapping=approved,
    ).output_artifact
    assert mapped is not None
    return mapped


def test_engines_satisfy_protocols(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    assert isinstance(ContractDataCleaner(storage), DataCleaner)
    assert isinstance(DeterministicFeatureEngineer(storage), FeatureEngineer)


def test_clean_and_engineer_deterministic_flow(tmp_path: Path):
    storage = LocalDatasetArtifactStorage(tmp_path)
    contract, context = make_contract_and_context()
    csv_bytes = (
        b"batch_id,planned_qty,machine_speed,shift_code,waste_qty\n"
        b" B1 ,100.0,300.0, A ,0.0\n"
        b"B2,200.0,,B,5.0\n"
        b"B3,400.0,400.0,A,10.0\n"
    )
    mapped = map_csv(storage, context, contract, csv_bytes)

    cleaner = ContractDataCleaner(storage)
    clean_res = cleaner.clean(context=context, artifact=mapped, contract=contract)
    assert clean_res.status == ExecutionStatus.SUCCEEDED
    assert clean_res.output_artifact is not None
    cleaned = clean_res.output_artifact
    assert cleaned.metadata.state == DatasetState.CLEANED

    # Missing machine_speed must NOT be imputed before splitting, and zero waste is preserved.
    with storage.open_verified(
        factory_id=context.factory_id, reference=cleaned.reference
    ) as stream:
        cleaned_df = pd.read_csv(stream)
    assert cleaned_df.loc[0, "batch_id"] == "B1"
    assert cleaned_df.loc[0, "shift_code"] == "A"
    assert cleaned_df.loc[0, "waste_qty"] == 0.0
    assert pd.isna(cleaned_df.loc[1, "machine_speed"])

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda frame: frame["machine_speed"] / frame["planned_qty"],
    )
    engineer = DeterministicFeatureEngineer(storage, derived_specs=(spec,))
    eng_res = engineer.engineer(context=context, artifact=cleaned, contract=contract)
    assert eng_res.status == ExecutionStatus.SUCCEEDED
    assert eng_res.output_artifact is not None
    engineered = eng_res.output_artifact
    assert engineered.metadata.state == DatasetState.ENGINEERED
    assert "speed_per_kg" in [c.name for c in engineered.metadata.columns]

    with storage.open_verified(
        factory_id=context.factory_id, reference=engineered.reference
    ) as stream:
        eng_df = pd.read_csv(stream)
    assert eng_df.loc[0, "speed_per_kg"] == pytest.approx(3.0)
    assert pd.isna(eng_df.loc[1, "speed_per_kg"])


def test_tabular_preprocessor_fits_on_train_only():
    contract, _ = make_contract_and_context()
    preprocessor = TabularPreprocessor(contract)

    train_df = pd.DataFrame(
        {
            "batch_id": ["B1", "B2"],
            "planned_qty": [100.0, 200.0],
            "machine_speed": [300.0, 500.0],
            "shift_code": ["A", "B"],
            "speed_per_kg": [3.0, 2.5],
            "waste_qty": [2.0, 4.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "shift_code": ["C", np.nan],
            "speed_per_kg": [np.nan, 4.0],
            "machine_speed": [np.nan, 9999.0],
            "planned_qty": [150.0, 250.0],
        }
    )

    with pytest.raises(RuntimeError, match="must be fit"):
        preprocessor.transform(test_df)

    x_train = preprocessor.fit_transform(train_df)
    assert x_train.shape[0] == 2
    assert not np.isnan(x_train).any()

    # Extreme test value (9999.0) must NOT change imputation/scaling stats learned from train.
    x_test = preprocessor.transform(test_df)
    assert x_test.shape == (2, x_train.shape[1])
    assert not np.isnan(x_test).any()
    # Unseen category "C" in test is safely ignored (all zero one-hot columns for shift_code).
    assert "shift_code_A" in preprocessor.get_feature_names_out()
    assert "shift_code_C" not in preprocessor.get_feature_names_out()
