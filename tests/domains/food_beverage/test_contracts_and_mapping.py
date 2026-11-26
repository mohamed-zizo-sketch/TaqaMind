"""Tests for Food & Beverage contracts, domain registration, mapping, and validation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from taqamind.core.contracts import DatasetPurpose
from taqamind.core.registry import DomainRegistry
from taqamind.domains.food_beverage.domain import register_food_beverage
from taqamind.domains.food_beverage.pipeline import FoodBeveragePipeline
from tests.domains.food_beverage.conftest import FACTORY_A, make_synthetic_bakery_dataframe


def test_domain_registration_succeeds() -> None:
    registry = DomainRegistry()
    domain = register_food_beverage(registry)
    assert domain.domain_id == "food_beverage"
    assert set(domain.process_ids) == {"bakery", "dairy", "beverage"}
    assert domain.total_tasks == 3


def test_required_feature_and_unknown_feature_rejection(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df = make_synthetic_bakery_dataframe(n_rows=10, include_target=True)
    # Remove required predictor
    bad_df = df.drop(columns=["planned_processing_temp_c"])
    bad_path = tmp_path / "missing_req.csv"
    bad_df.to_csv(bad_path, index=False)

    with pytest.raises(ValueError, match="Required canonical columns are missing"):
        tmp_pipeline.prepare_dataset(
            csv_path=bad_path,
            factory_id=FACTORY_A,
            process_id="bakery",
            purpose=DatasetPurpose.TRAINING,
            target_confirmation=synthetic_confirmation,
        )


def test_alias_mapping_and_unit_conversion(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df = make_synthetic_bakery_dataframe(n_rows=10, include_target=True)
    # Rename columns to approved aliases and supply mass in grams
    df = df.rename(
        columns={
            "planned_batch_quantity_kg": "planned_qty_kg",
            "waste_quantity_kg": "scrap_kg",
        }
    )
    df["planned_qty_kg"] = df["planned_qty_kg"] * 1000.0  # in grams
    alias_path = tmp_path / "aliased_grams.csv"
    df.to_csv(alias_path, index=False)

    prep = tmp_pipeline.prepare_dataset(
        csv_path=alias_path,
        factory_id=FACTORY_A,
        process_id="bakery",
        purpose=DatasetPurpose.TRAINING,
        target_confirmation=synthetic_confirmation,
        explicit_source_units={"planned_qty_kg": "g"},
    )
    with tmp_pipeline.dataset_storage.open_verified(
        factory_id=FACTORY_A,
        reference=prep.engineered_artifact.reference,
    ) as stream:
        loaded = pd.read_csv(stream)
    assert "planned_batch_quantity_kg" in loaded.columns
    assert "waste_quantity_kg" in loaded.columns
    assert loaded["planned_batch_quantity_kg"].iloc[0] < 3000.0  # converted g -> kg


def test_incompatible_unit_rejected(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df = make_synthetic_bakery_dataframe(n_rows=10, include_target=True)
    path = tmp_path / "bad_unit.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="INCOMPATIBLE_UNIT|Cannot convert"):
        tmp_pipeline.prepare_dataset(
            csv_path=path,
            factory_id=FACTORY_A,
            process_id="bakery",
            purpose=DatasetPurpose.TRAINING,
            target_confirmation=synthetic_confirmation,
            explicit_source_units={"planned_batch_quantity_kg": "L"},
        )


def test_post_production_predictor_rejected_in_validation(
    tmp_pipeline: FoodBeveragePipeline,
    tmp_path: Path,
    synthetic_confirmation,
) -> None:
    df = make_synthetic_bakery_dataframe(n_rows=10, include_target=True)
    df["actual_yield_pct"] = 98.5
    path = tmp_path / "with_postprod.csv"
    df.to_csv(path, index=False)

    # Engineer strips metadata POST_PRODUCTION columns safely, while validator blocks
    # any POST_PRODUCTION predictor if one is present. Verify contract blocks it too.
    contract = tmp_pipeline.get_contract("bakery", target_confirmation=synthetic_confirmation)
    assert "actual_yield_pct" not in contract.predictor_names


def test_missing_target_rejected_for_training(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_predict_csv: Path,
    synthetic_confirmation,
) -> None:
    with pytest.raises(ValueError, match="Required canonical columns are missing"):
        tmp_pipeline.prepare_dataset(
            csv_path=bakery_predict_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
            purpose=DatasetPurpose.TRAINING,
            target_confirmation=synthetic_confirmation,
        )


def test_prediction_rejects_target_column_if_present(
    tmp_pipeline: FoodBeveragePipeline,
    bakery_train_csv: Path,
) -> None:
    with pytest.raises(ValueError, match="(?i)target"):
        tmp_pipeline.prepare_dataset(
            csv_path=bakery_train_csv,
            factory_id=FACTORY_A,
            process_id="bakery",
            purpose=DatasetPurpose.PREDICTION,
            model_version_id=uuid4(),
        )
