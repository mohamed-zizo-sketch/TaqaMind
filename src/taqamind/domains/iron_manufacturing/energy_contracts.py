"""V2 contracts and derived feature specs for Iron Manufacturing electrical energy prediction."""

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
from taqamind.domains.iron_manufacturing.common.energy_formulas import (
    compute_theoretical_nameplate_energy_kwh,
)
from taqamind.domains.iron_manufacturing.config import DOMAIN_ID
from taqamind.domains.iron_manufacturing.processes.eaf_meltshop import (
    EAF_PROCESS_ID,
    build_eaf_derived_feature_specs,
)
from taqamind.domains.iron_manufacturing.processes.rebar_hot_rolling import (
    REBAR_PROCESS_ID,
)
from taqamind.domains.iron_manufacturing.processes.scrap_yard_reception import (
    SCRAP_YARD_PROCESS_ID,
)

EAF_ENERGY_TASK_ID = "eaf_electrical_energy_regression"
EAF_ENERGY_DATASET_ID = "eaf_meltshop_heat_energy"

REBAR_ENERGY_TASK_ID = "rebar_electrical_energy_regression"
REBAR_ENERGY_DATASET_ID = "rebar_rolling_lot_energy"

SCRAP_YARD_ENERGY_TASK_ID = "scrap_yard_electrical_energy_regression"
SCRAP_YARD_ENERGY_DATASET_ID = "scrap_yard_lot_energy"

ENERGY_PROCESS_TASK_AND_DATASET: dict[str, tuple[str, str]] = {
    EAF_PROCESS_ID: (EAF_ENERGY_TASK_ID, EAF_ENERGY_DATASET_ID),
    REBAR_PROCESS_ID: (REBAR_ENERGY_TASK_ID, REBAR_ENERGY_DATASET_ID),
    SCRAP_YARD_PROCESS_ID: (SCRAP_YARD_ENERGY_TASK_ID, SCRAP_YARD_ENERGY_DATASET_ID),
}

IRON_ENERGY_POST_PRODUCTION_EXCLUDED_COLUMNS: frozenset[str] = frozenset(
    {
        "active_power_kw",
        "reactive_power_kvar",
        "actual_peak_demand_kw",
        "actual_power_factor",
        "actual_tap_to_tap_min",
        "actual_power_on_min",
        "actual_power_off_delay_min",
        "in_process_arc_power_mw",
        "actual_final_temp_c",
        "actual_liquid_steel_tapped_t",
        "actual_good_rebar_output_t",
        "natural_gas_nm3",
        "reheat_gas_nm3",
        "thermal_energy_gj",
        "co2_emissions_ton",
        "actual_specific_energy_kwh_per_t",
    }
)


def build_synthetic_iron_energy_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for automated engineering verification."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="synthetic_iron_energy_test_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic engineering verification confirmation only; not a real factory sign-off.",
    )


def _shared_energy_envelope_features() -> tuple[FeatureContract, ...]:
    """Return shared pre-production electrical rating & derived envelope features."""
    rated_kw = FeatureContract(
        name="rated_kw",
        display_name="Equipment Rated Active Power (kW)",
        description="Nameplate active electrical power rating (kW) of the furnace or mill drive.",
        data_type=DataType.FLOAT,
        unit="kW",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=True,
        nullable=False,
        constraints=FeatureConstraints(minimum=1.0, maximum=250000.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("nameplate_kw", "rated_power_kw", "القدرة الاسمية"),
    )
    planned_duration_hr = FeatureContract(
        name="planned_duration_hr",
        display_name="Planned Operating Duration (hr)",
        description="Scheduled operating or tap-to-tap duration in hours prior to execution.",
        data_type=DataType.FLOAT,
        unit="hr",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        required=True,
        nullable=False,
        constraints=FeatureConstraints(minimum=0.05, maximum=48.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("scheduled_duration_hr", "planned_hours", "مدة التشغيل المخططة"),
    )
    planned_utilization_pct = FeatureContract(
        name="planned_utilization_pct",
        display_name="Planned Utilization (%)",
        description="Expected pre-production power utilization percentage.",
        data_type=DataType.FLOAT,
        unit="%",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=5.0, maximum=100.0),
        missing_policy=MissingValuePolicy.IMPUTE,
        aliases=("utilization_pct", "planned_load_pct", "نسبة التحميل المخططة"),
    )
    theoretical_nameplate_energy_kwh = FeatureContract(
        name="theoretical_nameplate_energy_kwh",
        display_name="Theoretical Nameplate Energy (kWh)",
        description=(
            "Physics-informed baseline: rated_kw * planned_duration_hr * (utilization / 100)."
        ),
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.IMPUTE,
        derived_from=("rated_kw", "planned_duration_hr", "planned_utilization_pct"),
    )
    return (
        rated_kw,
        planned_duration_hr,
        planned_utilization_pct,
        theoretical_nameplate_energy_kwh,
    )


def _build_energy_target_contract(
    *,
    task_id: str,
    granularity: str,
    definition_text: str,
    measurement_method: str,
    included_components: tuple[str, ...],
    excluded_components: tuple[str, ...],
    confirmation: TargetConfirmation | None = None,
) -> TargetContract:
    """Build an UNCONFIRMED-by-default electrical energy TargetContract."""
    target_feature = FeatureContract(
        name="total_electrical_energy_kwh",
        display_name="Total Electrical Energy Consumption (kWh)",
        description=(
            "Measured active electrical energy consumed during the confirmed production event."
        ),
        data_type=DataType.FLOAT,
        unit="kWh",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        required=True,
        nullable=False,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("electrical_energy_kwh", "energy_kwh", "استهلاك الكهرباء"),
    )
    return TargetContract(
        target_id=task_id,
        version="2.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=definition_text,
            granularity=granularity,
            measurement_method=measurement_method,
            included_components=included_components,
            excluded_components=excluded_components,
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="electrical_energy_to_nameplate_ratio",
            numerator="total_electrical_energy_kwh",
            denominator="theoretical_nameplate_energy_kwh",
            granularity=granularity,
            output_unit="ratio",
            multiplier=1.0,
        ),
    )


def build_eaf_energy_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build EAF Meltshop V2 pre-production electrical energy DatasetContract."""
    base_features: tuple[FeatureContract, ...] = (
        FeatureContract(
            name="heat_id",
            display_name="Heat Identifier",
            description="Unique business identifier for one EAF melting heat.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="production_order_id",
            display_name="Production Order Identifier",
            description="Campaign or customer production order grouping sequential heats.",
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_start_ts",
            display_name="Planned Heat Start Timestamp",
            description="Scheduled heat start timestamp prior to scrap basket charging (t0).",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="furnace_id",
            display_name="EAF Vessel Identifier",
            description="Physical electric arc furnace vessel identifier.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.STATIC,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="steel_grade_code",
            display_name="Target Steel Grade",
            description="Ordered steel metallurgy specification (e.g., B500DWR).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_scrap_mass_t",
            display_name="Planned Scrap Charge Mass (t)",
            description="Planned total ferrous scrap mass loaded into charging baskets.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.0, maximum=300.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_dri_mass_t",
            display_name="Planned DRI / HBI Charge Mass (t)",
            description="Planned Direct Reduced Iron or HBI mass scheduled for the heat.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.0, maximum=300.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_tap_mass_t",
            display_name="Planned Tap Mass (t)",
            description="Target liquid steel tap weight ordered for the heat.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.1, maximum=350.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="target_tap_temp_c",
            display_name="Target Tapping Temperature (degC)",
            description="Ordered liquid steel tapping temperature specification.",
            data_type=DataType.FLOAT,
            unit="degC",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=1520.0, maximum=1720.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_total_metallic_charge_t",
            display_name="Planned Total Metallic Charge (t)",
            description="Deterministic sum of planned_scrap_mass_t and planned_dri_mass_t.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.1, maximum=500.0),
            missing_policy=MissingValuePolicy.REJECT,
            derived_from=("planned_scrap_mass_t", "planned_dri_mass_t"),
        ),
        FeatureContract(
            name="planned_dri_share_pct",
            display_name="Planned DRI Charge Share (%)",
            description="Percentage of planned metallic charge supplied by DRI/HBI.",
            data_type=DataType.FLOAT,
            unit="%",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            required=False,
            nullable=True,
            constraints=FeatureConstraints(minimum=0.0, maximum=100.0),
            missing_policy=MissingValuePolicy.IMPUTE,
            derived_from=("planned_dri_mass_t", "planned_total_metallic_charge_t"),
        ),
    )

    target = _build_energy_target_contract(
        task_id=EAF_ENERGY_TASK_ID,
        granularity="one_eaf_heat",
        definition_text=(
            "Total active electrical energy (kWh) metered at the EAF furnace transformer "
            "and dedicated fume-extraction feeder per tap-to-tap heat."
        ),
        measurement_method="Sub-metered HV/MV active energy integrator per heat boundary.",
        included_components=("eaf_main_transformer", "eaf_dedusting_fan_feeder"),
        excluded_components=(
            "natural_gas_oxy_fuel_burners",
            "oxygen_plant_air_separation",
            "office_hvac",
        ),
        confirmation=confirmation,
    )

    return DatasetContract(
        dataset_id=EAF_ENERGY_DATASET_ID,
        version="2.0.0",
        domain=DOMAIN_ID,
        process=EAF_PROCESS_ID,
        description="Pre-production EAF meltshop heat dataset for electrical energy prediction.",
        granularity="one_eaf_heat",
        features=base_features + _shared_energy_envelope_features(),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict EAF electrical energy (kWh) prior to basket charging (t <= t0).",
        ),
        business_keys=("heat_id",),
        grouping_columns=("production_order_id",),
        time_column="planned_start_ts",
    )


def build_rebar_energy_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build Rebar Hot Rolling V2 pre-production electrical energy DatasetContract."""
    base_features: tuple[FeatureContract, ...] = (
        FeatureContract(
            name="rolling_lot_id",
            display_name="Rolling Lot Identifier",
            description="Unique business identifier for one billet-to-rebar rolling campaign lot.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="production_order_id",
            display_name="Production Order Identifier",
            description="Customer or rolling campaign order grouping sequential lots.",
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_start_ts",
            display_name="Planned Rolling Start Timestamp",
            description="Scheduled start timestamp prior to reheating furnace charging (t0).",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="mill_line_id",
            display_name="Rolling Mill Line Identifier",
            description="Physical rebar hot rolling mill line identifier.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.STATIC,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_billet_mass_t",
            display_name="Planned Billet Charge Mass (t)",
            description="Total scheduled billet mass entering the rolling campaign lot.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.1, maximum=5000.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="target_rebar_diameter_mm",
            display_name="Target Rebar Diameter (mm)",
            description="Ordered nominal deformed rebar diameter (e.g., 10, 12, 16, 25, 32 mm).",
            data_type=DataType.FLOAT,
            unit="mm",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=6.0, maximum=50.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
    )

    target = _build_energy_target_contract(
        task_id=REBAR_ENERGY_TASK_ID,
        granularity="one_rolling_campaign_lot",
        definition_text=(
            "Total active electrical energy (kWh) consumed by main rolling mill stand drives "
            "and reheating furnace electrical auxiliaries per rolling campaign lot."
        ),
        measurement_method="Sub-metered mill drive transformers + furnace auxiliary MCC feeders.",
        included_components=(
            "mill_stand_main_drives",
            "reheating_furnace_electrical_auxiliaries",
            "water_cooling_pumps",
        ),
        excluded_components=("reheating_furnace_natural_gas_thermal", "meltshop_eaf_feeders"),
        confirmation=confirmation,
    )

    return DatasetContract(
        dataset_id=REBAR_ENERGY_DATASET_ID,
        version="2.0.0",
        domain=DOMAIN_ID,
        process=REBAR_PROCESS_ID,
        description=(
            "Pre-production rebar hot rolling lot dataset for electrical energy prediction."
        ),
        granularity="one_rolling_campaign_lot",
        features=base_features + _shared_energy_envelope_features(),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict rolling lot electrical energy (kWh) prior to billet charging.",
        ),
        business_keys=("rolling_lot_id",),
        grouping_columns=("production_order_id",),
        time_column="planned_start_ts",
    )


def build_scrap_yard_energy_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build Scrap Yard Reception V2 pre-production electrical energy DatasetContract."""
    base_features: tuple[FeatureContract, ...] = (
        FeatureContract(
            name="reception_lot_id",
            display_name="Scrap Reception Lot Identifier",
            description="Unique business identifier for one scrap yard processing batch.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="delivery_batch_id",
            display_name="Delivery Batch Identifier",
            description="Supplier delivery campaign identifier for group-safe splitting.",
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_start_ts",
            display_name="Planned Processing Start Timestamp",
            description="Scheduled start timestamp prior to crane/shear/baler operation.",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
        ),
        FeatureContract(
            name="planned_scrap_mass_t",
            display_name="Planned Scrap Mass (t)",
            description="Scheduled ferrous scrap mass to be handled/sheared/baled.",
            data_type=DataType.FLOAT,
            unit="metric_ton",
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            constraints=FeatureConstraints(minimum=0.1, maximum=2000.0),
            missing_policy=MissingValuePolicy.REJECT,
        ),
    )

    target = _build_energy_target_contract(
        task_id=SCRAP_YARD_ENERGY_TASK_ID,
        granularity="one_scrap_reception_lot",
        definition_text=(
            "Total active electrical energy (kWh) consumed by magnetic cranes, hydraulic shears, "
            "and scrap balers per scrap processing batch."
        ),
        measurement_method="Sub-metered scrap yard substation feeder.",
        included_components=("magnetic_cranes", "hydraulic_shears", "scrap_balers"),
        excluded_components=("diesel_mobile_loaders", "meltshop_eaf_transformer"),
        confirmation=confirmation,
    )

    return DatasetContract(
        dataset_id=SCRAP_YARD_ENERGY_DATASET_ID,
        version="2.0.0",
        domain=DOMAIN_ID,
        process=SCRAP_YARD_PROCESS_ID,
        description="Pre-production scrap yard lot dataset for electrical energy prediction.",
        granularity="one_scrap_reception_lot",
        features=base_features + _shared_energy_envelope_features(),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict scrap yard electrical energy (kWh) prior to batch handling.",
        ),
        business_keys=("reception_lot_id",),
        grouping_columns=("delivery_batch_id",),
        time_column="planned_start_ts",
    )


def get_iron_energy_dataset_contract(
    process_id: str,
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Resolve the V2 Energy DatasetContract for a supported iron_manufacturing process."""
    if process_id == EAF_PROCESS_ID:
        return build_eaf_energy_dataset_contract(confirmation=confirmation)
    if process_id == REBAR_PROCESS_ID:
        return build_rebar_energy_dataset_contract(confirmation=confirmation)
    if process_id == SCRAP_YARD_PROCESS_ID:
        return build_scrap_yard_energy_dataset_contract(confirmation=confirmation)
    raise ValueError(f"Unsupported iron_manufacturing process '{process_id}'.")


def build_iron_energy_task_contract(
    process_id: str,
    *,
    confirmation: TargetConfirmation | None = None,
) -> IndustrialTaskContract:
    """Build the V2 Energy IndustrialTaskContract under PlatformModule.ENERGY_EFFICIENCY."""
    dataset_contract = get_iron_energy_dataset_contract(process_id, confirmation=confirmation)
    task_id, _ = ENERGY_PROCESS_TASK_AND_DATASET[process_id]
    return IndustrialTaskContract(
        task_id=task_id,
        version="2.0.0",
        display_name=f"Pre-Production {process_id} Electrical Energy Regression",
        description=f"Forecast expected electrical energy consumption (kWh) for {process_id}.",
        module=PlatformModule.ENERGY_EFFICIENCY,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )


def build_iron_energy_derived_specs(process_id: str) -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs reusing V1 specs + physics envelope."""
    nameplate_spec = DerivedFeatureSpec(
        feature_name="theoretical_nameplate_energy_kwh",
        compute=lambda df: compute_theoretical_nameplate_energy_kwh(
            df["rated_kw"],
            df["planned_duration_hr"],
            df.get("planned_utilization_pct"),
        ),
    )
    if process_id == EAF_PROCESS_ID:
        # Reuse compatible V1 EAF charge-composition derived features without duplication
        reused_eaf = tuple(
            spec
            for spec in build_eaf_derived_feature_specs()
            if spec.feature_name in {"planned_total_metallic_charge_t", "planned_dri_share_pct"}
        )
        return reused_eaf + (nameplate_spec,)
    if process_id == REBAR_PROCESS_ID:
        return (nameplate_spec,)
    if process_id == SCRAP_YARD_PROCESS_ID:
        return (nameplate_spec,)
    raise ValueError(f"Unsupported iron_manufacturing process '{process_id}'.")
