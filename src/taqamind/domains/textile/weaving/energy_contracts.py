"""Weaving V2 contracts for pre-production electrical energy consumption regression."""

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
    compute_theoretical_nameplate_energy_kwh,
    compute_weaving_thread_density_sum,
)
from taqamind.domains.textile.weaving.contracts import WEAVING_PROCESS_ID

WEAVING_ENERGY_TASK_ID = "weaving_energy_regression"
WEAVING_ENERGY_DATASET_ID = "weaving_energy_batches"

WEAVING_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "actual_operating_duration_hr",
    "actual_produced_length_m",
    "actual_machine_efficiency_pct",
    "actual_stop_count",
    "actual_downtime_hr",
    "actual_compressed_air_nm3",
    "actual_energy_intensity_kwh_per_m",
)


def build_synthetic_weaving_energy_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for automated engineering verification."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="synthetic_textile_energy_test_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic engineering verification confirmation only; not a real factory sign-off.",
    )


def build_weaving_energy_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Weaving V2 Energy DatasetContract (UNCONFIRMED by default)."""
    batch_record_id = FeatureContract(
        name="batch_record_id",
        display_name="Beam Batch Record ID",
        description="Unique weaving beam batch or loom-run record identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    order_id = FeatureContract(
        name="order_id",
        display_name="Production Order ID",
        description="Production planning order identifier used for group-safe splitting.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    scheduled_date = FeatureContract(
        name="scheduled_date",
        display_name="Scheduled Production Date",
        description="Scheduled calendar date of the weaving beam batch.",
        data_type=DataType.DATE,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_id = FeatureContract(
        name="machine_id",
        display_name="Loom Machine ID",
        description="Unique loom identifier within the factory.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        missing_policy=MissingValuePolicy.REJECT,
    )
    loom_type = FeatureContract(
        name="loom_type",
        display_name="Loom Technology Type",
        description="Weft insertion technology (air_jet, rapier, water_jet).",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(allowed_values=("air_jet", "rapier", "water_jet")),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_rated_power_kw = FeatureContract(
        name="machine_rated_power_kw",
        display_name="Loom Rated Power (kW)",
        description="Nameplate electrical motor rating (kW), not actual consumed kWh.",
        data_type=DataType.FLOAT,
        unit="kW",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_beam_length_m = FeatureContract(
        name="planned_beam_length_m",
        display_name="Planned Beam Length (m)",
        description="Planned weaver beam length in meters prior to loom execution.",
        data_type=DataType.FLOAT,
        unit="m",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    planned_operating_duration_hr = FeatureContract(
        name="planned_operating_duration_hr",
        display_name="Planned Operating Duration (h)",
        description="Scheduled loom operating duration in hours.",
        data_type=DataType.FLOAT,
        unit="h",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.1),
        missing_policy=MissingValuePolicy.REJECT,
    )
    fabric_width_cm = FeatureContract(
        name="fabric_width_cm",
        display_name="Fabric Reed Width (cm)",
        description="Configured fabric/reed working width in centimeters.",
        data_type=DataType.FLOAT,
        unit="cm",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=30.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    planned_loom_speed_rpm = FeatureContract(
        name="planned_loom_speed_rpm",
        display_name="Planned Loom Speed (rpm)",
        description="Target loom speed in picks per minute (RPM).",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=50.0),
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
    compressed_air_pressure_bar = FeatureContract(
        name="compressed_air_pressure_bar",
        display_name="Compressed Air Setpoint (bar)",
        description="Planned main/relay nozzle compressed air pressure in bar.",
        data_type=DataType.FLOAT,
        unit="bar",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=0.0, maximum=12.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    warp_count_ne = FeatureContract(
        name="warp_count_ne",
        display_name="Warp Yarn Count (Ne)",
        description="English cotton count of longitudinal warp yarn.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    weft_count_ne = FeatureContract(
        name="weft_count_ne",
        display_name="Weft Yarn Count (Ne)",
        description="English cotton count of transverse weft yarn.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    warp_ends_per_inch = FeatureContract(
        name="warp_ends_per_inch",
        display_name="Warp Ends Per Inch (EPI)",
        description="Number of warp threads per inch in fabric specification.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    weft_picks_per_inch = FeatureContract(
        name="weft_picks_per_inch",
        display_name="Weft Picks Per Inch (PPI)",
        description="Number of weft picks per inch in fabric specification.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    ambient_temp_c = FeatureContract(
        name="ambient_temp_c",
        display_name="Ambient Shed Temperature (degC)",
        description="Planned/target weaving shed ambient temperature in Celsius.",
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
    relative_humidity_pct = FeatureContract(
        name="relative_humidity_pct",
        display_name="Relative Humidity Setpoint (%)",
        description="Planned weaving shed relative humidity percentage.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=20.0, maximum=95.0),
        missing_policy=MissingValuePolicy.IMPUTE,
    )
    total_thread_density_per_inch = FeatureContract(
        name="total_thread_density_per_inch",
        display_name="Total Thread Density (EPI + PPI)",
        description="Derived sum of warp EPI and weft PPI from pre-production inputs.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=2.0),
        derived_from=("warp_ends_per_inch", "weft_picks_per_inch"),
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

    total_electrical_energy_kwh = FeatureContract(
        name="total_electrical_energy_kwh",
        display_name="Total Electrical Energy Consumption (kWh)",
        description="Measured electrical energy consumed by the loom batch (machine + air eq.).",
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )

    target = TargetContract(
        target_id=WEAVING_ENERGY_TASK_ID,
        feature=total_electrical_energy_kwh,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured electrical energy consumption (kWh) per weaving beam batch at "
                "machine-run boundary (including allocated compressed air where sub-metered)."
            ),
            granularity="beam_batch",
            measurement_method="Sub-metered loom electrical power integrator + compressor share.",
            included_components=("main_loom_drive", "auxiliary_controls", "compressed_air_share"),
            excluded_components=("whole_factory_hvac", "sizing_steam_thermal", "office_lighting"),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="weaving_energy_to_nameplate_ratio",
            numerator="total_electrical_energy_kwh",
            denominator="theoretical_nameplate_energy_kwh",
            granularity="beam_batch",
            output_unit="ratio",
            multiplier=1.0,
        ),
    )

    return DatasetContract(
        dataset_id=WEAVING_ENERGY_DATASET_ID,
        version="1.0.0",
        domain="textile",
        process=WEAVING_PROCESS_ID,
        description=(
            "Pre-production weaving electrical energy consumption prediction contract at "
            "beam-batch granularity with order grouping and scheduled date."
        ),
        granularity="beam_batch",
        features=(
            batch_record_id,
            order_id,
            scheduled_date,
            machine_id,
            loom_type,
            machine_rated_power_kw,
            planned_beam_length_m,
            planned_operating_duration_hr,
            fabric_width_cm,
            planned_loom_speed_rpm,
            planned_utilization_pct,
            compressed_air_pressure_bar,
            warp_count_ne,
            weft_count_ne,
            warp_ends_per_inch,
            weft_picks_per_inch,
            ambient_temp_c,
            relative_humidity_pct,
            total_thread_density_per_inch,
            theoretical_nameplate_energy_kwh,
        ),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict weaving electrical energy (kWh) prior to beam mounting.",
        ),
        business_keys=("batch_record_id",),
        grouping_columns=("order_id",),
        time_column="scheduled_date",
    )


def build_weaving_energy_derived_specs() -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs for Weaving Energy V2."""
    return (
        DerivedFeatureSpec(
            feature_name="total_thread_density_per_inch",
            compute=lambda df: compute_weaving_thread_density_sum(
                df["warp_ends_per_inch"], df["weft_picks_per_inch"]
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


def build_weaving_energy_task_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the Weaving Energy IndustrialTaskContract."""
    dataset_contract = build_weaving_energy_dataset_contract(confirmation=confirmation)
    return IndustrialTaskContract(
        task_id=WEAVING_ENERGY_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Weaving Electrical Energy Regression",
        description="Forecast expected electrical energy consumption (kWh) per weaving batch.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
