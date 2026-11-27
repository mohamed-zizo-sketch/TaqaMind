"""Tests for Food & Beverage V2 Energy contracts, Core integration, and zero-retraining."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from taqamind.core.automl.trainer import ControlledAutoMLTrainer, TabularPreprocessor
from taqamind.core.contracts import ConfirmationStatus, DatasetPurpose, PlatformModule
from taqamind.core.interfaces import (
    AcceptancePolicy,
    EvaluationDecision,
    ModelStatus,
    PredictionStatus,
    TrainingMode,
)
from taqamind.core.registry import DomainRegistry
from taqamind.domains.food_beverage.domain import register_food_beverage
from taqamind.domains.food_beverage.energy_contracts import (
    BAKERY_ENERGY_TASK_ID,
    BEVERAGE_ENERGY_TASK_ID,
    DAIRY_ENERGY_TASK_ID,
    FOOD_BEVERAGE_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS,
    build_bakery_energy_dataset_contract,
    build_beverage_energy_dataset_contract,
    build_dairy_energy_dataset_contract,
    build_synthetic_energy_confirmation,
)
from taqamind.domains.food_beverage.energy_pipeline import FoodBeverageEnergyPipelineService
from taqamind.domains.food_beverage.energy_synthetic import (
    generate_synthetic_bakery_energy_prediction_frame,
    generate_synthetic_bakery_energy_training_frame,
    generate_synthetic_beverage_energy_prediction_frame,
    generate_synthetic_beverage_energy_training_frame,
    generate_synthetic_dairy_energy_prediction_frame,
    generate_synthetic_dairy_energy_training_frame,
)


def test_food_beverage_v2_energy_domain_registration_and_unconfirmed_default() -> None:
    reg = DomainRegistry()
    domain = register_food_beverage(reg)
    expected_tasks = {
        "bakery": BAKERY_ENERGY_TASK_ID,
        "dairy": DAIRY_ENERGY_TASK_ID,
        "beverage": BEVERAGE_ENERGY_TASK_ID,
    }
    for proc_id, task_id in expected_tasks.items():
        proc = domain.get_process(proc_id)
        assert task_id in proc.task_ids
        energy_task = proc.get_task(task_id)
        assert energy_task.module == PlatformModule.ENERGY_EFFICIENCY
        assert (
            energy_task.dataset_contract.target.confirmation.status
            == ConfirmationStatus.UNCONFIRMED
        )
        for col in FOOD_BEVERAGE_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS:
            assert col not in energy_task.dataset_contract.predictor_names


def test_unconfirmed_energy_target_blocks_training(short_tmp_dir: Path) -> None:
    service = FoodBeverageEnergyPipelineService(short_tmp_dir / "s")
    factory_id = uuid4()
    contract = build_bakery_energy_dataset_contract(confirmation=None)
    df = generate_synthetic_bakery_energy_training_frame(num_lines=12, batches_per_line=2)
    csv_path = short_tmp_dir / "unconfirmed.csv"
    df.to_csv(csv_path, index=False)

    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="bakery",
        csv_path=csv_path,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
    )
    with pytest.raises(ValueError, match="not eligible"):
        service.train_and_evaluate(
            factory_id=factory_id,
            process_id="bakery",
            prepared=prepared,
            contract=contract,
        )


@pytest.mark.parametrize("process_id", ["bakery", "dairy", "beverage"])
def test_v2_energy_e2e_core_platform_and_zero_retraining_prediction(
    short_tmp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_id: str,
) -> None:
    service = FoodBeverageEnergyPipelineService(short_tmp_dir / process_id)
    factory_id = uuid4()
    created_at = datetime.now(UTC) - timedelta(minutes=10)
    conf = build_synthetic_energy_confirmation()

    if process_id == "bakery":
        contract = build_bakery_energy_dataset_contract(confirmation=conf)
        train_df = generate_synthetic_bakery_energy_training_frame(
            num_lines=20, batches_per_line=3, seed=42
        )
        pred_df = generate_synthetic_bakery_energy_prediction_frame(num_batches=6, seed=99)
    elif process_id == "dairy":
        contract = build_dairy_energy_dataset_contract(confirmation=conf)
        train_df = generate_synthetic_dairy_energy_training_frame(
            num_lines=20, batches_per_line=3, seed=42
        )
        pred_df = generate_synthetic_dairy_energy_prediction_frame(num_batches=6, seed=88)
    else:
        contract = build_beverage_energy_dataset_contract(confirmation=conf)
        train_df = generate_synthetic_beverage_energy_training_frame(
            num_lines=20, batches_per_line=3, seed=42
        )
        pred_df = generate_synthetic_beverage_energy_prediction_frame(num_batches=6, seed=77)

    train_csv = short_tmp_dir / f"{process_id}_energy_train.csv"
    pred_csv = short_tmp_dir / f"{process_id}_energy_pred.csv"
    train_df.to_csv(train_csv, index=False)
    pred_df.to_csv(pred_csv, index=False)

    prepared = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id=process_id,
        csv_path=train_csv,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=created_at,
    )
    policy = AcceptancePolicy(
        minimum_improvement_pct=1.0,
        justification="V2 Energy E2E test.",
        created_at=created_at - timedelta(minutes=5),
    )
    outcome = service.train_and_evaluate(
        factory_id=factory_id,
        process_id=process_id,
        prepared=prepared,
        contract=contract,
        policy=policy,
    )
    assert outcome.evaluation.decision == EvaluationDecision.ACCEPTED
    assert outcome.activated
    assert outcome.model_entry.status == ModelStatus.ACTIVE

    # Zero-retraining spy across 3 predictions + fresh service reload
    def _forbid_fit(*args, **kwargs):
        raise AssertionError("Training/fitting must NEVER occur during prediction!")

    monkeypatch.setattr(TabularPreprocessor, "fit", _forbid_fit)
    monkeypatch.setattr(ControlledAutoMLTrainer, "train", _forbid_fit)
    monkeypatch.setattr(Ridge, "fit", _forbid_fit)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", _forbid_fit)

    reloaded = FoodBeverageEnergyPipelineService(short_tmp_dir / process_id)
    for _ in range(3):
        batch = reloaded.predict(
            factory_id=factory_id,
            process_id=process_id,
            csv_path=pred_csv,
            contract=contract,
        )
        assert batch.successful_count == len(pred_df)
        assert all(p.status == PredictionStatus.SUCCEEDED for p in batch.predictions)
        assert all(
            p.predicted_value is not None and p.predicted_value >= 0.0
            for p in batch.predictions
        )
        assert all(
            p.derived_rate is not None and p.derived_rate.value is not None
            for p in batch.predictions
        )

    # Factory B isolation check
    factory_b = uuid4()
    with pytest.raises(ValueError, match="No active energy model"):
        reloaded.predict(
            factory_id=factory_b,
            process_id=process_id,
            csv_path=pred_csv,
            contract=contract,
        )


def test_v2_energy_explicit_retraining_and_target_leakage_guard(short_tmp_dir: Path) -> None:
    service = FoodBeverageEnergyPipelineService(short_tmp_dir / "s")
    factory_id = uuid4()
    contract = build_dairy_energy_dataset_contract(
        confirmation=build_synthetic_energy_confirmation()
    )
    t0 = datetime.now(UTC) - timedelta(hours=1)
    train_df1 = generate_synthetic_dairy_energy_training_frame(
        num_lines=18, batches_per_line=3, seed=10
    )
    csv1 = short_tmp_dir / "d_train1.csv"
    train_df1.to_csv(csv1, index=False)

    prep1 = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="dairy",
        csv_path=csv1,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=t0,
    )
    out1 = service.train_and_evaluate(
        factory_id=factory_id,
        process_id="dairy",
        prepared=prep1,
        contract=contract,
        policy=AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="v1",
            created_at=t0 - timedelta(minutes=5),
        ),
    )
    assert out1.activated

    # Prediction CSV containing target column must be rejected!
    with pytest.raises(ValueError, match="must not contain target column"):
        service.predict(
            factory_id=factory_id,
            process_id="dairy",
            csv_path=csv1,
            contract=contract,
        )

    # Explicit retraining on new data promotes v2 and archives v1
    train_df2 = generate_synthetic_dairy_energy_training_frame(
        num_lines=18, batches_per_line=3, seed=20
    )
    train_df2["batch_id"] = "R2_" + train_df2["batch_id"].astype(str)
    train_df2["production_line_id"] = "L2_" + train_df2["production_line_id"].astype(str)
    csv2 = short_tmp_dir / "d_train2.csv"
    train_df2.to_csv(csv2, index=False)

    t1 = datetime.now(UTC) - timedelta(minutes=5)
    prep2 = service.ingest_and_prepare(
        factory_id=factory_id,
        process_id="dairy",
        csv_path=csv2,
        purpose=DatasetPurpose.TRAINING,
        contract=contract,
        created_at=t1,
    )
    out2 = service.train_and_evaluate(
        factory_id=factory_id,
        process_id="dairy",
        prepared=prep2,
        contract=contract,
        policy=AcceptancePolicy(
            minimum_improvement_pct=1.0,
            justification="v2",
            created_at=t1 - timedelta(minutes=5),
        ),
        mode=TrainingMode.RETRAINING,
        previous_model_version_id=out1.model_entry.version.model_version_id,
    )
    assert out2.activated
    assert out2.model_entry.version.version_number == 2
    assert out2.model_entry.status == ModelStatus.ACTIVE
    assert (
        service.core.model_registry.get(
            scope=out1.model_entry.version.scope,
            model_version_id=out1.model_entry.version.model_version_id,
        ).status
        == ModelStatus.ARCHIVED
    )
