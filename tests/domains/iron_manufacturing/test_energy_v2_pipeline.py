"""End-to-end engineering verification tests for Iron Manufacturing V2 Electrical Energy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from taqamind.core.automl.trainer import ControlledAutoMLTrainer, TabularPreprocessor
from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetPurpose,
    PlatformModule,
)
from taqamind.core.interfaces import (
    AcceptancePolicy,
    EvaluationDecision,
    ModelScope,
    ModelStatus,
    PredictionStatus,
    TrainingMode,
)
from taqamind.domains.iron_manufacturing import (
    DOMAIN_ID,
    EAF_ENERGY_TASK_ID,
    IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS,
    REBAR_ENERGY_TASK_ID,
    SCRAP_YARD_ENERGY_TASK_ID,
    IronEnergyPipelineService,
    build_iron_manufacturing_domain,
    build_synthetic_iron_energy_confirmation,
    generate_energy_process_templates,
    generate_synthetic_iron_energy_dataset,
    get_iron_energy_dataset_contract,
)
from taqamind.domains.iron_manufacturing.cli import main as cli_main
from taqamind.domains.iron_manufacturing.processes.eaf_meltshop import (
    EAF_PROCESS_ID,
    EAF_TASK_ID,
)
from taqamind.domains.iron_manufacturing.processes.rebar_hot_rolling import (
    REBAR_PROCESS_ID,
    REBAR_TASK_ID,
)
from taqamind.domains.iron_manufacturing.processes.scrap_yard_reception import (
    SCRAP_YARD_PROCESS_ID,
    SCRAP_YARD_TASK_ID,
)

ENERGY_PROCESS_TASK_PAIRS = (
    (EAF_PROCESS_ID, EAF_TASK_ID, EAF_ENERGY_TASK_ID),
    (REBAR_PROCESS_ID, REBAR_TASK_ID, REBAR_ENERGY_TASK_ID),
    (SCRAP_YARD_PROCESS_ID, SCRAP_YARD_TASK_ID, SCRAP_YARD_ENERGY_TASK_ID),
)


@pytest.mark.parametrize(
    ("process_id", "waste_task_id", "energy_task_id"),
    ENERGY_PROCESS_TASK_PAIRS,
)
def test_v2_energy_contracts_separated_and_unconfirmed_by_default(
    process_id: str,
    waste_task_id: str,
    energy_task_id: str,
) -> None:
    domain = build_iron_manufacturing_domain()
    waste_task = domain.get_task(process_id=process_id, task_id=waste_task_id)
    energy_task = domain.get_task(process_id=process_id, task_id=energy_task_id)

    assert waste_task.module == PlatformModule.WASTE_MANAGEMENT
    assert energy_task.module == PlatformModule.ENERGY_EFFICIENCY
    assert (
        energy_task.dataset_contract.target.confirmation.status
        == ConfirmationStatus.UNCONFIRMED
    )
    assert not energy_task.dataset_contract.target.is_confirmed
    assert energy_task.dataset_contract.target.feature.name == "total_electrical_energy_kwh"
    assert energy_task.dataset_contract.target.feature.unit == "kWh"

    for col in IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS:
        assert col not in energy_task.dataset_contract.predictor_names


def test_unconfirmed_energy_target_blocks_training(
    tmp_path: Path, storage_root: Path
) -> None:
    factory_id = uuid4()
    service = IronEnergyPipelineService(storage_root)
    unconfirmed_contract = get_iron_energy_dataset_contract(EAF_PROCESS_ID)
    train_csv = tmp_path / "eaf_train.csv"
    generate_synthetic_iron_energy_dataset(EAF_PROCESS_ID, n_rows=60, seed=42).to_csv(
        train_csv, index=False
    )

    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=unconfirmed_contract,
    )
    with pytest.raises(ValueError, match="not eligible"):
        service.train_and_evaluate(
            factory_id=factory_id,
            process_id=EAF_PROCESS_ID,
            prepared=prepared,
            contract=unconfirmed_contract,
        )


def test_leakage_columns_rejected_on_ingest_and_prediction(
    tmp_path: Path, storage_root: Path
) -> None:
    factory_id = uuid4()
    service = IronEnergyPipelineService(storage_root)
    contract = get_iron_energy_dataset_contract(
        EAF_PROCESS_ID,
        confirmation=build_synthetic_iron_energy_confirmation(),
    )

    leaked_train = tmp_path / "leaked_train.csv"
    generate_synthetic_iron_energy_dataset(
        EAF_PROCESS_ID,
        n_rows=40,
        seed=42,
        leak_column="actual_peak_demand_kw",
    ).to_csv(leaked_train, index=False)

    with pytest.raises(ValueError, match="forbidden post-production"):
        service.ingest_and_prepare(
            factory_id=factory_id,
            process_id=EAF_PROCESS_ID,
            csv_path=leaked_train,
            purpose=DatasetPurpose.TRAINING,
            contract=contract,
        )

    train_csv = tmp_path / "eaf_train.csv"
    generate_synthetic_iron_energy_dataset(EAF_PROCESS_ID, n_rows=80, seed=42).to_csv(
        train_csv, index=False
    )
    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    outcome = service.train_and_evaluate(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        prepared=prepared,
        contract=contract,
    )
    assert outcome.activated is True

    pred_with_target = tmp_path / "pred_with_target.csv"
    generate_synthetic_iron_energy_dataset(
        EAF_PROCESS_ID, n_rows=10, seed=99, include_target=True
    ).to_csv(pred_with_target, index=False)

    with pytest.raises(ValueError, match="must not contain target column"):
        service.predict(
            factory_id=factory_id,
            process_id=EAF_PROCESS_ID,
            csv_path=pred_with_target,
            contract=contract,
        )


@pytest.mark.parametrize(
    "process_id",
    [EAF_PROCESS_ID, REBAR_PROCESS_ID, SCRAP_YARD_PROCESS_ID],
)
def test_end_to_end_energy_lifecycle_zero_retraining_and_factory_isolation(
    tmp_path: Path,
    storage_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_id: str,
) -> None:
    factory_a = uuid4()
    factory_b = uuid4()
    store_dir = storage_root
    service = IronEnergyPipelineService(store_dir)
    confirmation = build_synthetic_iron_energy_confirmation()
    contract = get_iron_energy_dataset_contract(process_id, confirmation=confirmation)

    created_at = datetime.now(UTC) - timedelta(minutes=10)
    train_csv = tmp_path / f"{process_id}_train.csv"
    generate_synthetic_iron_energy_dataset(process_id, n_rows=80, seed=11).to_csv(
        train_csv, index=False
    )

    prepared = service.ingest_and_prepare(
        factory_id=factory_a,
        process_id=process_id,
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=created_at,
    )
    policy = AcceptancePolicy(
        minimum_improvement_pct=1.0,
        justification=f"V2 {process_id} acceptance gate.",
        created_at=created_at - timedelta(minutes=5),
    )
    outcome = service.train_and_evaluate(
        factory_id=factory_a,
        process_id=process_id,
        prepared=prepared,
        contract=contract,
        policy=policy,
    )
    assert outcome.evaluation.decision == EvaluationDecision.ACCEPTED
    assert outcome.activated is True
    assert outcome.model_entry.status == ModelStatus.ACTIVE

    pred_csv = tmp_path / f"{process_id}_pred.csv"
    generate_synthetic_iron_energy_dataset(
        process_id,
        n_rows=10,
        seed=22,
        include_target=False,
        start_index=500,
    ).to_csv(pred_csv, index=False)

    # Zero-retraining guard across multiple predictions and fresh service reload
    def _forbid_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("Fitting/training must NEVER occur during prediction!")

    monkeypatch.setattr(TabularPreprocessor, "fit", _forbid_fit)
    monkeypatch.setattr(ControlledAutoMLTrainer, "train", _forbid_fit)
    monkeypatch.setattr(Ridge, "fit", _forbid_fit)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", _forbid_fit)

    reloaded = IronEnergyPipelineService(store_dir)
    for _ in range(2):
        preds = reloaded.predict(
            factory_id=factory_a,
            process_id=process_id,
            csv_path=pred_csv,
            contract=contract,
        )
        assert preds.successful_count == 10
        assert all(p.status == PredictionStatus.SUCCEEDED for p in preds.predictions)
        assert all(
            p.predicted_value is not None and p.predicted_value > 0.0
            for p in preds.predictions
        )
        assert all(
            p.derived_rate is not None and p.derived_rate.value is not None
            for p in preds.predictions
        )

    # Multi-tenant factory isolation: Factory B must have no active model
    with pytest.raises(ValueError, match="No active energy model"):
        reloaded.predict(
            factory_id=factory_b,
            process_id=process_id,
            csv_path=pred_csv,
            contract=contract,
        )


def test_v2_energy_retraining_promotion_and_version_history(
    tmp_path: Path, storage_root: Path
) -> None:
    factory_id = uuid4()
    service = IronEnergyPipelineService(storage_root)
    contract = get_iron_energy_dataset_contract(
        EAF_PROCESS_ID,
        confirmation=build_synthetic_iron_energy_confirmation(),
    )

    t0 = datetime.now(UTC) - timedelta(hours=1)
    train_csv_v1 = tmp_path / "eaf_train_v1.csv"
    generate_synthetic_iron_energy_dataset(EAF_PROCESS_ID, n_rows=80, seed=11).to_csv(
        train_csv_v1, index=False
    )
    prep_v1 = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        csv_path=train_csv_v1,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=t0,
    )
    out_v1 = service.train_and_evaluate(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        prepared=prep_v1,
        contract=contract,
        policy=AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="v1 gate",
            created_at=t0 - timedelta(minutes=5),
        ),
    )
    assert out_v1.activated is True

    t1 = datetime.now(UTC) - timedelta(minutes=5)
    train_csv_v2 = tmp_path / "eaf_train_v2.csv"
    generate_synthetic_iron_energy_dataset(
        EAF_PROCESS_ID,
        n_rows=96,
        seed=44,
        start_index=1000,
        start_date="2026-06-01T06:00:00+00:00",
    ).to_csv(train_csv_v2, index=False)
    prep_v2 = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        csv_path=train_csv_v2,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=t1,
    )
    out_v2 = service.train_and_evaluate(
        factory_id=factory_id,
        process_id=EAF_PROCESS_ID,
        prepared=prep_v2,
        contract=contract,
        policy=AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="v2 gate",
            created_at=t1 - timedelta(minutes=5),
        ),
        mode=TrainingMode.RETRAINING,
        previous_model_version_id=out_v1.model_entry.version.model_version_id,
    )
    assert out_v2.activated is True
    assert out_v2.model_entry.version.version_number == 2
    assert out_v2.model_entry.status == ModelStatus.ACTIVE

    scope = ModelScope(
        factory_id=factory_id,
        domain=DOMAIN_ID,
        process=EAF_PROCESS_ID,
        task_id=EAF_ENERGY_TASK_ID,
    )
    versions = service.core.model_registry.list_versions(scope=scope)
    assert len(versions) == 2
    v1_after = service.core.model_registry.get(
        scope=scope,
        model_version_id=out_v1.model_entry.version.model_version_id,
    )
    assert v1_after.status == ModelStatus.ARCHIVED


def test_v2_energy_templates_and_cli_parity(
    tmp_path: Path, storage_root: Path
) -> None:
    for process_id in (EAF_PROCESS_ID, REBAR_PROCESS_ID, SCRAP_YARD_PROCESS_ID):
        files = generate_energy_process_templates(process_id, tmp_path / "tpl" / process_id)
        hist_df = pd.read_csv(files["historical_template"])
        pred_df = pd.read_csv(files["prediction_template"])
        assert "total_electrical_energy_kwh" in hist_df.columns
        assert "total_electrical_energy_kwh" not in pred_df.columns
        assert "theoretical_nameplate_energy_kwh" not in hist_df.columns
        assert len(hist_df) == 3
        assert len(pred_df) == 3

    store = storage_root
    factory_id = uuid4()
    train_csv = tmp_path / "scrap_train.csv"
    pred_csv = tmp_path / "scrap_pred.csv"
    out_csv = tmp_path / "scrap_pred_out.csv"
    generate_synthetic_iron_energy_dataset(SCRAP_YARD_PROCESS_ID, n_rows=80, seed=55).to_csv(
        train_csv, index=False
    )
    generate_synthetic_iron_energy_dataset(
        SCRAP_YARD_PROCESS_ID,
        n_rows=6,
        seed=88,
        include_target=False,
        start_index=400,
    ).to_csv(pred_csv, index=False)

    assert (
        cli_main(
            [
                "--storage-root",
                str(store),
                "--module",
                "energy",
                "inspect",
                "--factory-id",
                str(factory_id),
                "--process",
                SCRAP_YARD_PROCESS_ID,
                "--csv",
                str(train_csv),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--storage-root",
                str(store),
                "--module",
                "energy",
                "train",
                "--factory-id",
                str(factory_id),
                "--process",
                SCRAP_YARD_PROCESS_ID,
                "--csv",
                str(train_csv),
                "--synthetic-confirmation",
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--storage-root",
                str(store),
                "--module",
                "energy",
                "predict",
                "--factory-id",
                str(factory_id),
                "--process",
                SCRAP_YARD_PROCESS_ID,
                "--csv",
                str(pred_csv),
                "--output-csv",
                str(out_csv),
                "--synthetic-confirmation",
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "--storage-root",
                str(store),
                "--module",
                "energy",
                "model-info",
                "--factory-id",
                str(factory_id),
                "--process",
                SCRAP_YARD_PROCESS_ID,
            ]
        )
        == 0
    )
    exported = pd.read_csv(out_csv)
    assert len(exported) == 6
    assert "predicted_value" in exported.columns
