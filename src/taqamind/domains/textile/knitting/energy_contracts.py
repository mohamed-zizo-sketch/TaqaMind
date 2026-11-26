"""Knitting V2 contracts for pre-production electrical energy consumption regression."""

from __future__ import annotations

from datetime import UTC, datetime

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
from taqamind.domains.textile.common.formulas import (
    compute_knitting_tightness_factor,
    compute_theoretical_nameplate_energy_kwh,
)
from taqamind.domains.textile.knitting.contracts import KNITTING_PROCESS_ID

KNITTING_ENERGY_TASK_ID = "knitting_energy_regression"
KNITTING_ENERGY_DATASET_ID = "knitting_energy_runs"

KNITTING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "actual_operating_duration_hr",
    "actual_produced_qty_kg",
    "actual_machine_efficiency_pct",
    "actual_stop_count",
    "actual_downtime_hr",
    "actual_energy_intensity_kwh_per_kg",
)


def build_synthetic_knitting_energy_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for automated engineering verification."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="synthetic_textile_energy_test_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic engineering verification confirmation only; not a real factory sign-off.",
    )


def build_knitting_energy_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Knitting V2 Energy DatasetContract (UNCONFIRMED by default)."""
    run_id = FeatureContract(
        name="run_id",
        display_name="Knitting Run ID",
        description="Unique knitting shift-run production identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    order_lot_id = FeatureContract(
        name="order_lot_id",
        display_name="Order Lot Identifier",
        description="Knitting lot or customer order grouping identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    production_date = FeatureContract(
        name="production_date",
        display_name="Scheduled Production Date",
        description="Scheduled calendar date of the knitting run.",
        data_type=DataType.DATE,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_id = FeatureContract(
        name="machine_id",
        display_name="Knitting Machine ID",
        description="Unique knitting machine identifier within the factory.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    knitting_machine_type = FeatureContract(
        name="knitting_machine_type",
        display_name="Knitting Machine Type",
        description="Machine architecture (circular or flat).",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(allowed_values=("circular", "flat")),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_rated_power_kw = FeatureContract(
        name="machine_rated_power_kw",
        display_name="Machine Rated Power (kW)",
        description="Nameplate electrical drive rating (kW), not actual consumed kWh.",
        data_type=DataType.FLOAT,
        unit="kW",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_qty_kg = FeatureContract(
        name="planned_qty_kg",
        display_name="Planned Production Quantity (kg)",
        description="Planned fabric/yarn mass allocated for the knitting shift run.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_operating_duration_hr = FeatureContract(
        name="planned_operating_duration_hr",
        display_name="Planned Operating Duration (h)",
        description="Scheduled knitting machine operating duration in hours.",
        data_type=DataType.FLOAT,
        unit="h",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_diameter_in = FeatureContract(
        name="machine_diameter_in",
        display_name="Machine Cylinder Diameter (in)",
        description="Circular knitting cylinder diameter in inches.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_gauge_e = FeatureContract(
        name="machine_gauge_e",
        display_name="Machine Gauge (E)",
        description="Needles per inch (machine gauge).",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    number_of_feeders = FeatureContract(
        name="number_of_feeders",
        display_name="Active Yarn Feeders Count",
        description="Number of active yarn feeds configured for the knit structure.",
        data_type=DataType.INTEGER,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=1),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    planned_machine_speed_rpm = FeatureContract(
        name="planned_machine_speed_rpm",
        display_name="Planned Machine Speed (rpm)",
        description="Target cylinder/carriage speed in RPM.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=5.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    planned_utilization_pct = FeatureContract(
        name="planned_utilization_pct",
        display_name="Planned Utilization (%)",
        description="Expected pre-run machine utilization percentage.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=10.0, maximum=100.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    yarn_count_tex_1 = FeatureContract(
        name="yarn_count_tex_1",
        display_name="Primary Yarn Linear Density (tex)",
        description="Primary feeder yarn linear density in tex.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    stitch_length_mm = FeatureContract(
        name="stitch_length_mm",
        display_name="Planned Stitch Length (mm)",
        description="Target loop length per stitch in millimeters.",
        data_type=DataType.FLOAT,
        unit="mm",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=0.5),
        missing_policy=MissingValuePolicy.REJECT,
    )
    target_gsm = FeatureContract(
        name="target_gsm",
        display_name="Target Fabric GSM (g/m2)",
        description="Specified target fabric areal weight in g/m2.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=20.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    ambient_temp_c = FeatureContract(
        name="ambient_temp_c",
        display_name="Ambient Shed Temperature (degC)",
        description="Planned knitting shed ambient temperature in Celsius.",
        data_type=DataType.FLOAT,
        unit="degC",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=5.0, maximum=55.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    knitting_tightness_factor = FeatureContract(
        name="knitting_tightness_factor",
        display_name="Knitting Tightness Factor (K)",
        description="Derived Munden tightness factor sqrt(tex) / (stitch_length_mm / 10).",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.IMPUTE,
        derived_from=("yarn_count_tex_1", "stitch_length_mm"),
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
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.IMPUTE,
        derived_from=(
            "machine_rated_power_kw",
            "planned_operating_duration_hr",
            "planned_utilization_pct",
        ),
    )

    total_electrical_energy_kwh = FeatureContract(
        name="total_electrical_energy_kwh",
        display_name="Total Electrical Energy Consumption (kWh)",
        description="Measured electrical energy consumed by the knitting machine during run.",
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )

    target = TargetContract(
        target_id=KNITTING_ENERGY_TASK_ID,
        feature=total_electrical_energy_kwh,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured electrical energy consumption (kWh) per knitting shift run at "
                "machine-run boundary (main drive + positive yarn feeders + control panel)."
            ),
            granularity="shift_run",
            measurement_method="Sub-metered knitting machine electrical energy meter (kWh).",
            included_components=("main_drive_motor", "positive_feeders", "control_electronics"),
            excluded_components=("whole_factory_hvac", "central_lighting", "dyeing_steam"),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="knitting_energy_to_nameplate_ratio",
            numerator="total_electrical_energy_kwh",
            denominator="theoretical_nameplate_energy_kwh",
            granularity="shift_run",
            output_unit="ratio",
            multiplier=1.0,
        ),
    )

    return DatasetContract(
        dataset_id=KNITTING_ENERGY_DATASET_ID,
        version="1.0.0",
        domain="textile",
        process=KNITTING_PROCESS_ID,
        description=(
            "Pre-production knitting electrical energy consumption prediction contract at "
            "shift-run granularity with lot grouping and scheduled date."
        ),
        granularity="shift_run",
        features=(
            run_id,
            order_lot_id,
            production_date,
            machine_id,
            knitting_machine_type,
            machine_rated_power_kw,
            planned_qty_kg,
            planned_operating_duration_hr,
            machine_diameter_in,
            machine_gauge_e,
            number_of_feeders,
            planned_machine_speed_rpm,
            planned_utilization_pct,
            yarn_count_tex_1,
            stitch_length_mm,
            target_gsm,
            ambient_temp_c,
            knitting_tightness_factor,
            theoretical_nameplate_energy_kwh,
        ),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict knitting electrical energy (kWh) before starting shift run.",
        ),
        business_keys=("run_id",),
        grouping_columns=("order_lot_id",),
        time_column="production_date",
    )


def build_knitting_energy_derived_specs() -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs for Knitting Energy V2."""
    return (
        DerivedFeatureSpec(
            feature_name="knitting_tightness_factor",
            compute=lambda df: compute_knitting_tightness_factor(
                df["yarn_count_tex_1"], df["stitch_length_mm"]
            ),
        ),
        DerivedFeatureSpec(
            feature_name="theoretical_nameplate_energy_kwh",
            compute=lambda df: compute_theoretical_nameplate_energy_kwh(
                df["machine_rated_power_kw"],
                df["planned_operating_duration_hr"],
                df.get("planned_utilization_pct"),
            ),
        ),
    )


def build_knitting_energy_task_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the Knitting Energy IndustrialTaskContract."""
    dataset_contract = build_knitting_energy_dataset_contract(confirmation=confirmation)
    return IndustrialTaskContract(
        task_id=KNITTING_ENERGY_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Knitting Electrical Energy Regression",
        description="Forecast expected electrical energy consumption (kWh) per knitting shift run.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
