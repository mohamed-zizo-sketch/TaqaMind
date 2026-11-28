"""Contract, registration, protocol conformance, mapping, unit, and validation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from taqamind.core.automl import (
    ControlledAutoMLTrainer,
    LocalDevelopmentDataProvider,
    LocalEvaluationRegistry,
    LocalProtectedTestDataProvider,
    LocalTrainingRegistry,
)
from taqamind.core.contracts import ConfirmationStatus, DatasetPurpose
from taqamind.core.feature_engineering import DeterministicFeatureEngineer
from taqamind.core.ingestion import CsvDataIngestion
from taqamind.core.interfaces import (
    ActiveModelProvider,
    ColumnMapper,
    DataCleaner,
    DataIngestion,
    DataProfiler,
    DatasetArtifactStorage,
    DatasetRegistry,
    DatasetSplitter,
    DevelopmentDataProvider,
    EvaluationRegistry,
    FeatureEngineer,
    LeakageValidator,
    ModelArtifactStorage,
    ModelRegistry,
    ModelTrainer,
    PredictionEngine,
    ProtectedTestDataProvider,
    RecordLineageRegistry,
    RecordUsageRegistry,
    SemanticValidator,
    SplitRegistry,
    TrainingReadinessChecker,
    TrainingRegistry,
)
from taqamind.core.lineage import LocalRecordLineageRegistry
from taqamind.core.pipeline import CorePlatformServices
from taqamind.core.prediction import CorePredictionEngine, LocalActiveModelProvider
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.profiling import PandasDataProfiler
from taqamind.core.readiness import CoreTrainingReadinessChecker
from taqamind.core.registry import (
    DomainRegistry,
    LocalDatasetArtifactStorage,
    LocalDatasetRegistry,
    LocalModelArtifactStorage,
    LocalModelRegistry,
)
from taqamind.core.splitting import (
    CoreDatasetSplitter,
    CoreLeakageValidator,
    LocalRecordUsageRegistry,
    LocalSplitRegistry,
)
from taqamind.domains.iron_manufacturing import (
    DOMAIN_ID,
    DOMAIN_VERSION,
    IronManufacturingPipeline,
    generate_process_templates,
    get_iron_dataset_contract,
    register_iron_manufacturing,
)
from taqamind.domains.iron_manufacturing.common.synthetic import (
    generate_synthetic_eaf_dataset,
)
from taqamind.domains.iron_manufacturing.common.units import (
    UnitConversionError,
    convert_series_unit,
)
from tests.domains.iron_manufacturing.conftest import FACTORY_A_ID


def test_domain_registration_and_default_unconfirmed_targets() -> None:
    registry = DomainRegistry()
    domain = register_iron_manufacturing(registry)

    assert domain.domain_id == DOMAIN_ID
    assert domain.version == DOMAIN_VERSION
    assert registry.has_domain(DOMAIN_ID, DOMAIN_VERSION)
    assert set(domain.process_ids) == {
        "eaf_meltshop",
        "rebar_hot_rolling",
        "scrap_yard_reception",
    }

    # Idempotent re-registration succeeds without error
    again = register_iron_manufacturing(registry)
    assert again == domain

    # All default production targets (both V1 Waste and V2 Energy) remain UNCONFIRMED
    for process_id in domain.process_ids:
        process = domain.get_process(process_id)
        assert len(process.tasks) == 2
        for task in process.tasks:
            assert (
                task.dataset_contract.target.confirmation.status
                == ConfirmationStatus.UNCONFIRMED
            )
            assert not task.dataset_contract.target.is_confirmed
        contract = get_iron_dataset_contract(process_id)
        assert contract.target.confirmation.status == ConfirmationStatus.UNCONFIRMED
        assert not contract.target.is_confirmed


def test_all_local_adapters_satisfy_shared_core_protocols(
    pipeline: IronManufacturingPipeline,
) -> None:
    assert isinstance(pipeline.dataset_storage, DatasetArtifactStorage)
    assert isinstance(pipeline.model_storage, ModelArtifactStorage)
    assert isinstance(pipeline.dataset_registry, DatasetRegistry)
    assert isinstance(pipeline.lineage_registry, RecordLineageRegistry)
    assert isinstance(pipeline.split_registry, SplitRegistry)
    assert isinstance(pipeline.usage_registry, RecordUsageRegistry)
    assert isinstance(pipeline.training_registry, TrainingRegistry)
    assert isinstance(pipeline.evaluation_registry, EvaluationRegistry)
    assert isinstance(pipeline.model_registry, ModelRegistry)
    assert isinstance(pipeline.dev_data_provider, DevelopmentDataProvider)
    assert isinstance(pipeline.protected_test_provider, ProtectedTestDataProvider)
    assert isinstance(pipeline.active_model_provider, ActiveModelProvider)
    assert isinstance(pipeline.ingestion, DataIngestion)
    assert isinstance(pipeline.profiler, DataProfiler)
    assert isinstance(pipeline.column_mapper, ColumnMapper)
    assert isinstance(pipeline.semantic_validator, SemanticValidator)
    assert isinstance(pipeline.data_cleaner, DataCleaner)
    assert isinstance(pipeline.feature_engineer, FeatureEngineer)
    assert isinstance(pipeline.readiness_checker, TrainingReadinessChecker)
    assert isinstance(pipeline.splitter, DatasetSplitter)
    assert isinstance(pipeline.leakage_validator, LeakageValidator)
    assert isinstance(pipeline.model_trainer, ModelTrainer)
    assert isinstance(pipeline.prediction_engine, PredictionEngine)

    # Verify real Shared Core P1.1-P1.8 concrete engines are wired via CorePlatformServices
    assert isinstance(pipeline.core, CorePlatformServices)
    assert isinstance(pipeline.dataset_storage, LocalDatasetArtifactStorage)
    assert isinstance(pipeline.model_storage, LocalModelArtifactStorage)
    assert isinstance(pipeline.dataset_registry, LocalDatasetRegistry)
    assert isinstance(pipeline.lineage_registry, LocalRecordLineageRegistry)
    assert isinstance(pipeline.split_registry, LocalSplitRegistry)
    assert isinstance(pipeline.usage_registry, LocalRecordUsageRegistry)
    assert isinstance(pipeline.training_registry, LocalTrainingRegistry)
    assert isinstance(pipeline.evaluation_registry, LocalEvaluationRegistry)
    assert isinstance(pipeline.model_registry, LocalModelRegistry)
    assert isinstance(pipeline.dev_data_provider, LocalDevelopmentDataProvider)
    assert isinstance(pipeline.protected_test_provider, LocalProtectedTestDataProvider)
    assert isinstance(pipeline.active_model_provider, LocalActiveModelProvider)
    assert isinstance(pipeline.ingestion, CsvDataIngestion)
    assert isinstance(pipeline.profiler, PandasDataProfiler)
    assert isinstance(pipeline.data_cleaner, ContractDataCleaner)
    assert isinstance(pipeline.readiness_checker, CoreTrainingReadinessChecker)
    assert isinstance(pipeline.splitter, CoreDatasetSplitter)
    assert isinstance(pipeline.leakage_validator, CoreLeakageValidator)
    assert isinstance(pipeline.model_trainer, ControlledAutoMLTrainer)
    assert isinstance(pipeline.prediction_engine, CorePredictionEngine)
    for proc in ("eaf_meltshop", "rebar_hot_rolling", "scrap_yard_reception"):
        assert isinstance(
            pipeline.feature_engineer.get_engineer(proc), DeterministicFeatureEngineer
        )


def test_templates_separate_training_and_prediction_schemas(tmp_path: Path) -> None:
    for process_id in ("eaf_meltshop", "rebar_hot_rolling", "scrap_yard_reception"):
        files = generate_process_templates(process_id, tmp_path / process_id)
        hist_df = pd.read_csv(files["historical_template"])
        pred_df = pd.read_csv(files["prediction_template"])
        contract = get_iron_dataset_contract(process_id)
        target_name = contract.target.feature.name

        assert target_name in hist_df.columns
        assert target_name not in pred_df.columns
        assert len(hist_df) == 3
        assert len(pred_df) == 3


def test_ambiguous_header_is_never_blindly_mapped(
    pipeline: IronManufacturingPipeline, tmp_path: Path
) -> None:
    df = generate_synthetic_eaf_dataset(n_rows=10, seed=1, include_target=True)
    df = df.rename(columns={"planned_scrap_mass_t": "Weight"})
    csv_path = tmp_path / "ambiguous_weight.csv"
    df.to_csv(csv_path, index=False)

    inspection = pipeline.inspect_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=csv_path,
    )
    assert "Weight" in inspection["mapping_proposal"]["unmapped_columns"]

    val_res = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        confirm_all_proposed_mappings=True,
    )
    assert val_res["status"] == "blocked"
    assert val_res["stage"] == "column_mapping"


def test_valid_and_invalid_unit_conversions(tmp_path: Path, storage_root: Path) -> None:
    s = pd.Series([1000.0, 2500.0])
    converted = convert_series_unit(s, "kg", "metric_ton")
    assert converted.tolist() == [1.0, 2.5]

    # Dimensionally incompatible, unknown, count-to-mass, and mass-to-length conversions fail
    for bad_src, bad_dst in (
        ("kg", "degC"),
        ("unknown_unit", "metric_ton"),
        ("billets", "kg"),
        ("heats", "metric_ton"),
        ("kg", "m"),
        ("m", "metric_ton"),
        ("kWh", "kg"),
    ):
        with pytest.raises(UnitConversionError):
            convert_series_unit(s, bad_src, bad_dst)

    # End-to-end invalid unit override is blocked cleanly in column mapping
    df = generate_synthetic_eaf_dataset(n_rows=10, seed=2, include_target=True)
    csv_path = tmp_path / "bad_unit.csv"
    df.to_csv(csv_path, index=False)

    bad_unit_pipeline = IronManufacturingPipeline(
        storage_root=storage_root,
        unit_overrides={"planned_scrap_mass_t": "degC"},
    )
    val_res = bad_unit_pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        confirm_all_proposed_mappings=True,
    )
    assert val_res["status"] == "blocked"
    assert any(i["code"] == "INVALID_UNIT_CONVERSION" for i in val_res["issues"])


def test_missing_training_target_rejected_and_prediction_without_target_passes(
    pipeline: IronManufacturingPipeline, tmp_path: Path
) -> None:
    df_no_target = generate_synthetic_eaf_dataset(n_rows=10, seed=3, include_target=False)
    no_target_csv = tmp_path / "no_target.csv"
    df_no_target.to_csv(no_target_csv, index=False)

    train_val = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=no_target_csv,
        purpose=DatasetPurpose.TRAINING,
    )
    assert train_val["status"] == "blocked"

    pred_val = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=no_target_csv,
        purpose=DatasetPurpose.PREDICTION,
    )
    assert pred_val["status"] == "ready"

    # Missing target value inside labeled CSV must NEVER be imputed
    df_nan_target = generate_synthetic_eaf_dataset(n_rows=10, seed=4, include_target=True)
    df_nan_target.loc[2, "metallic_waste_loss_t"] = None
    nan_csv = tmp_path / "nan_target.csv"
    df_nan_target.to_csv(nan_csv, index=False)

    nan_val = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=nan_csv,
        purpose=DatasetPurpose.TRAINING,
    )
    assert nan_val["status"] == "blocked"
    assert any(
        i["code"] == "MISSING_VALUES_REJECTED" and "metallic_waste_loss_t" in i["columns"]
        for i in nan_val["issues"]
    )


def test_post_production_predictor_rejected_and_zero_waste_preserved(
    pipeline: IronManufacturingPipeline, tmp_path: Path
) -> None:
    # Prediction CSV containing target column must be rejected
    df_with_target = generate_synthetic_eaf_dataset(n_rows=10, seed=5, include_target=True)
    with_target_csv = tmp_path / "pred_with_target.csv"
    df_with_target.to_csv(with_target_csv, index=False)

    res_target = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=with_target_csv,
        purpose=DatasetPurpose.PREDICTION,
    )
    assert res_target["status"] == "blocked"

    # Valid zero-scrap rows must pass validation cleanly without dropping or rejecting
    df_zeros = generate_synthetic_eaf_dataset(n_rows=12, seed=6, include_target=True)
    df_zeros.loc[0, "metallic_waste_loss_t"] = 0.0
    df_zeros.loc[1, "metallic_waste_loss_t"] = 0.0
    zeros_csv = tmp_path / "zeros_valid.csv"
    df_zeros.to_csv(zeros_csv, index=False)

    res_zeros = pipeline.validate_dataset(
        factory_id=FACTORY_A_ID,
        process_id="eaf_meltshop",
        csv_path=zeros_csv,
        purpose=DatasetPurpose.TRAINING,
    )
    assert res_zeros["status"] == "ready"
