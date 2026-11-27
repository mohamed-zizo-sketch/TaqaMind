"""Food & Beverage V2 contracts for pre-production electrical energy consumption regression."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from taqamind.core.contracts import (
    ConfirmationStatus,
    DatasetContract,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    IndustrialTaskContract,
    MissingValuePolicy,
    PlatformModule,
    PredictionContext,
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.feature_engineering import DerivedFeatureSpec

BAKERY_ENERGY_PROCESS_ID = "bakery"
BAKERY_ENERGY_TASK_ID = "bakery_energy_regression"
BAKERY_ENERGY_DATASET_ID = "bakery_energy_batches"

DAIRY_ENERGY_PROCESS_ID = "dairy"
DAIRY_ENERGY_TASK_ID = "dairy_energy_regression"
DAIRY_ENERGY_DATASET_ID = "dairy_energy_batches"

BEVERAGE_ENERGY_PROCESS_ID = "beverage"
BEVERAGE_ENERGY_TASK_ID = "beverage_energy_regression"
BEVERAGE_ENERGY_DATASET_ID = "beverage_energy_batches"

FOOD_BEVERAGE_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "actual_operating_duration_hr",
    "actual_produced_quantity_kg",
    "actual_downtime_min",
    "actual_stop_count",
    "actual_thermal_energy_mj",
    "actual_sec_kwh_per_kg",
)


def build_synthetic_energy_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for automated engineering verification."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="food_beverage_synthetic_energy_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic engineering verification confirmation only; not a real factory sign-off.",
    )


def compute_theoretical_nameplate_energy_kwh(
    rated_power_kw: pd.Series,
    planned_duration_hr: pd.Series,
    planned_utilization_pct: pd.Series | None = None,
) -> pd.Series:
    """Compute deterministic nameplate baseline electrical energy (kWh)."""
    util = (
        planned_utilization_pct.fillna(100.0) / 100.0
        if planned_utilization_pct is not None
        else 1.0
    )
    return rated_power_kw.astype(float) * planned_duration_hr.astype(float) * util


def build_energy_derived_specs() -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs for Food & Beverage Energy V2."""
    return (
        DerivedFeatureSpec(
            feature_name="theoretical_nameplate_energy_kwh",
            compute=lambda df: compute_theoretical_nameplate_energy_kwh(
                df["machine_rated_power_kw"],
                df["planned_operating_duration_hr"],
                df["planned_utilization_pct"] if "planned_utilization_pct" in df.columns else None,
            ),
        ),
    )


def _shared_energy_features() -> tuple[FeatureContract, ...]:
    batch_id = FeatureContract(
        name="batch_id",
        display_name="Batch ID",
        description="Unique production batch or shift-run identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    production_line_id = FeatureContract(
        name="production_line_id",
        display_name="Production Line ID",
        description="Production line identifier used for group-safe splitting.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    production_timestamp = FeatureContract(
        name="production_timestamp",
        display_name="Scheduled Production Date",
        description="Scheduled production date of the batch or shift run.",
        data_type=DataType.DATE,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_id = FeatureContract(
        name="machine_id",
        display_name="Machine ID",
        description="Unique equipment/machine identifier within the production line.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    product_category = FeatureContract(
        name="product_category",
        display_name="Product Category",
        description="Product family scheduled for the batch.",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_batch_quantity_kg = FeatureContract(
        name="planned_batch_quantity_kg",
        display_name="Planned Batch Quantity (kg)",
        description="Planned batch output mass in kilograms.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_rated_power_kw = FeatureContract(
        name="machine_rated_power_kw",
        display_name="Machine Rated Power (kW)",
        description="Nameplate electrical power rating of the primary machine/line.",
        data_type=DataType.FLOAT,
        unit="kW",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=0.5),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_operating_duration_hr = FeatureContract(
        name="planned_operating_duration_hr",
        display_name="Planned Operating Duration (hr)",
        description="Scheduled operating hours for the batch or shift run.",
        data_type=DataType.FLOAT,
        unit="h",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_utilization_pct = FeatureContract(
        name="planned_utilization_pct",
        display_name="Planned Utilization (%)",
        description="Expected capacity utilization percentage (0-100%).",
        data_type=DataType.FLOAT,
        unit="%",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=1.0, maximum=100.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    ambient_temp_c = FeatureContract(
        name="ambient_temp_c",
        display_name="Ambient Temperature (C)",
        description="Ambient plant temperature in degrees Celsius.",
        data_type=DataType.FLOAT,
        unit="C",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=-10.0, maximum=60.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    theoretical_nameplate_energy_kwh = FeatureContract(
        name="theoretical_nameplate_energy_kwh",
        display_name="Theoretical Nameplate Energy (kWh)",
        description="Physics-informed baseline: rated_kW * planned_hr * (utilization / 100).",
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=0.0),
        derived_from=(
            "machine_rated_power_kw",
            "planned_operating_duration_hr",
            "planned_utilization_pct",
        ),
    )
    return (
        batch_id,
        production_line_id,
        production_timestamp,
        machine_id,
        product_category,
        planned_batch_quantity_kg,
        machine_rated_power_kw,
        planned_operating_duration_hr,
        planned_utilization_pct,
        ambient_temp_c,
        theoretical_nameplate_energy_kwh,
    )


def _energy_target_contract(
    *,
    task_id: str,
    rate_name: str,
    definition_text: str,
    included_components: tuple[str, ...],
    excluded_components: tuple[str, ...],
    confirmation: TargetConfirmation | None,
) -> TargetContract:
    target_feature = FeatureContract(
        name="total_electrical_energy_kwh",
        display_name="Total Electrical Energy Consumption (kWh)",
        description="Measured electrical energy consumed by the production batch (kWh).",
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    return TargetContract(
        target_id=task_id,
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=definition_text,
            granularity="batch",
            measurement_method="Sub-metered line/machine electrical energy integrator (kWh).",
            included_components=included_components,
            excluded_components=excluded_components,
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name=rate_name,
            numerator="total_electrical_energy_kwh",
            denominator="theoretical_nameplate_energy_kwh",
            granularity="batch",
            output_unit="ratio",
            multiplier=1.0,
        ),
    )


def build_bakery_energy_dataset_contract(
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Bakery V2 Energy DatasetContract (UNCONFIRMED by default)."""
    oven_heating_type = FeatureContract(
        name="oven_heating_type",
        display_name="Oven Heating Type",
        description="Heating mode of the bakery line (electric vs gas_fired).",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_processing_temp_c = FeatureContract(
        name="planned_processing_temp_c",
        display_name="Planned Baking Temperature (C)",
        description="Target oven setpoint temperature in degrees Celsius.",
        data_type=DataType.FLOAT,
        unit="C",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=20.0, maximum=350.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    target = _energy_target_contract(
        task_id=BAKERY_ENERGY_TASK_ID,
        rate_name="bakery_energy_to_nameplate_ratio",
        definition_text=(
            "Measured electrical energy (kWh) per bakery batch across mixers, provers, "
            "electric oven elements/fans, cooling conveyors, and packaging."
        ),
        included_components=(
            "dough_mixers",
            "dividers_moulders",
            "provers",
            "electric_oven_load_or_fans",
            "cooling_conveyors",
            "packaging",
        ),
        excluded_components=("natural_gas_thermal_mj", "office_hvac", "facility_lighting"),
        confirmation=confirmation,
    )
    return DatasetContract(
        dataset_id=BAKERY_ENERGY_DATASET_ID,
        version="1.0.0",
        domain="food_beverage",
        process=BAKERY_ENERGY_PROCESS_ID,
        description="Pre-production bakery electrical energy consumption prediction contract.",
        granularity="batch",
        features=(*_shared_energy_features(), oven_heating_type, planned_processing_temp_c),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict bakery electrical energy (kWh) prior to batch start.",
        ),
        business_keys=("batch_id",),
        grouping_columns=("production_line_id",),
        time_column="production_timestamp",
    )


def build_dairy_energy_dataset_contract(
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Dairy V2 Energy DatasetContract (UNCONFIRMED by default)."""
    chiller_stage_type = FeatureContract(
        name="chiller_stage_type",
        display_name="Dairy Line Stage Type",
        description=(
            "Primary electrical load stage (pasteurization_chiller, uht_line, cold_storage)."
        ),
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_pasteurization_temp_c = FeatureContract(
        name="planned_pasteurization_temp_c",
        display_name="Planned Processing Temperature (C)",
        description="Target pasteurization or UHT setpoint temperature (C).",
        data_type=DataType.FLOAT,
        unit="C",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=2.0, maximum=150.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    target = _energy_target_contract(
        task_id=DAIRY_ENERGY_TASK_ID,
        rate_name="dairy_energy_to_nameplate_ratio",
        definition_text=(
            "Measured electrical energy (kWh) per dairy batch across refrigeration/chillers, "
            "homogenizers, separators, pasteurizer pumps, CIP pumps, and compressed air."
        ),
        included_components=(
            "refrigeration_chillers",
            "homogenizers",
            "separators",
            "pasteurizer_pumps",
            "cip_pumps",
            "compressed_air",
        ),
        excluded_components=("boiler_steam_thermal_mj", "office_hvac", "facility_lighting"),
        confirmation=confirmation,
    )
    return DatasetContract(
        dataset_id=DAIRY_ENERGY_DATASET_ID,
        version="1.0.0",
        domain="food_beverage",
        process=DAIRY_ENERGY_PROCESS_ID,
        description="Pre-production dairy electrical energy consumption prediction contract.",
        granularity="batch",
        features=(*_shared_energy_features(), chiller_stage_type, planned_pasteurization_temp_c),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict dairy electrical energy (kWh) prior to batch start.",
        ),
        business_keys=("batch_id",),
        grouping_columns=("production_line_id",),
        time_column="production_timestamp",
    )


def build_beverage_energy_dataset_contract(
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Beverage V2 Energy DatasetContract (UNCONFIRMED by default)."""
    packaging_line_type = FeatureContract(
        name="packaging_line_type",
        display_name="Packaging Line Type",
        description="Beverage line type (pet_bottling, canning, glass_bottling).",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_line_speed_bph = FeatureContract(
        name="planned_line_speed_bph",
        display_name="Planned Line Speed (bottles/hr)",
        description="Scheduled bottling or canning line speed in units per hour.",
        data_type=DataType.FLOAT,
        unit="bph",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=500.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    target = _energy_target_contract(
        task_id=BEVERAGE_ENERGY_TASK_ID,
        rate_name="beverage_energy_to_nameplate_ratio",
        definition_text=(
            "Measured electrical energy (kWh) per beverage batch across bottling/canning lines, "
            "conveyors, carbonation chillers, air compressors/PET blow-molding, and CIP."
        ),
        included_components=(
            "bottling_canning_drives",
            "conveyors",
            "carbonation_chillers",
            "air_compressors_pet_blow",
            "cip_pumps",
        ),
        excluded_components=("pasteurizer_steam_thermal_mj", "office_hvac", "facility_lighting"),
        confirmation=confirmation,
    )
    return DatasetContract(
        dataset_id=BEVERAGE_ENERGY_DATASET_ID,
        version="1.0.0",
        domain="food_beverage",
        process=BEVERAGE_ENERGY_PROCESS_ID,
        description="Pre-production beverage electrical energy consumption prediction contract.",
        granularity="batch",
        features=(*_shared_energy_features(), packaging_line_type, planned_line_speed_bph),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict beverage electrical energy (kWh) prior to batch start.",
        ),
        business_keys=("batch_id",),
        grouping_columns=("production_line_id",),
        time_column="production_timestamp",
    )


def build_bakery_energy_task_contract(
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the Bakery Energy IndustrialTaskContract."""
    return IndustrialTaskContract(
        task_id=BAKERY_ENERGY_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Bakery Electrical Energy Regression",
        description="Forecast expected electrical energy consumption (kWh) per bakery batch.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=build_bakery_energy_dataset_contract(confirmation=confirmation),
    )


def build_dairy_energy_task_contract(
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the Dairy Energy IndustrialTaskContract."""
    return IndustrialTaskContract(
        task_id=DAIRY_ENERGY_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Dairy Electrical Energy Regression",
        description="Forecast expected electrical energy consumption (kWh) per dairy batch.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=build_dairy_energy_dataset_contract(confirmation=confirmation),
    )


def build_beverage_energy_task_contract(
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the Beverage Energy IndustrialTaskContract."""
    return IndustrialTaskContract(
        task_id=BEVERAGE_ENERGY_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Beverage Electrical Energy Regression",
        description="Forecast expected electrical energy consumption (kWh) per beverage batch.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=build_beverage_energy_dataset_contract(confirmation=confirmation),
    )


__all__ = [
    "BAKERY_ENERGY_DATASET_ID",
    "BAKERY_ENERGY_PROCESS_ID",
    "BAKERY_ENERGY_TASK_ID",
    "BEVERAGE_ENERGY_DATASET_ID",
    "BEVERAGE_ENERGY_PROCESS_ID",
    "BEVERAGE_ENERGY_TASK_ID",
    "DAIRY_ENERGY_DATASET_ID",
    "DAIRY_ENERGY_PROCESS_ID",
    "DAIRY_ENERGY_TASK_ID",
    "FOOD_BEVERAGE_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS",
    "build_bakery_energy_dataset_contract",
    "build_bakery_energy_task_contract",
    "build_beverage_energy_dataset_contract",
    "build_beverage_energy_task_contract",
    "build_dairy_energy_dataset_contract",
    "build_dairy_energy_task_contract",
    "build_energy_derived_specs",
    "build_synthetic_energy_confirmation",
    "compute_theoretical_nameplate_energy_kwh",
]
