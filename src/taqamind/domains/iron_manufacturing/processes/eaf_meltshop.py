"""Electric Arc Furnace (EAF) primary meltshop process contract & rules.

Industrial scope:
    Egyptian EAF steelmaking (both integrated Midrex DRI+EAF meltshops and
    scrap-based mini-mills).
Row granularity:
    One furnace heat (`heat_id`).
Prediction timing:
    Strictly pre-production (`t <= t0`), prior to scrap basket charging and DRI
    feeding. In-process arc telemetry and post-tap weights are strictly excluded
    from predictors.
Target:
    `metallic_waste_loss_t` (metric_ton) — total metallic charge loss per heat.
    Unconfirmed by default until an Egyptian partner factory confirms hot-heel
    and ladle-skull accounting semantics.
"""

from __future__ import annotations

import numpy as np
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
    IssueImpact,
    MissingValuePolicy,
    PipelineStage,
    PlatformModule,
    PredictionContext,
    ProcessContract,
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
    ValidationIssue,
    ValidationSeverity,
)
from taqamind.domains.iron_manufacturing.config import DOMAIN_ID

EAF_PROCESS_ID = "eaf_meltshop"
EAF_TASK_ID = "eaf_metallic_waste_loss_prediction"
EAF_DATASET_ID = "eaf_meltshop_heat_waste"
EAF_GRANULARITY = "one_eaf_heat"

# Known post-production / in-process columns that must never be used as pre-production predictors.
EAF_FORBIDDEN_PREDICTOR_NAMES: frozenset[str] = frozenset(
    {
        "actual_total_metallic_charge_t",
        "actual_liquid_steel_tapped_t",
        "slag_mass_t",
        "eaf_dust_mass_t",
        "actual_electrical_energy_kwh",
        "actual_peak_demand_kw",
        "actual_power_factor",
        "actual_tap_to_tap_min",
        "actual_power_off_delay_min",
        "actual_power_on_min",
        "in_process_arc_power_mw",
        "actual_final_temp_c",
        "in_process_gas_purge_nm3",
        "in_process_bulk_additions_kg",
        "in_process_wire_additions_kg",
        "metallic_yield_pct",
    }
)


def build_eaf_meltshop_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the authoritative Shared-Core DatasetContract for EAF Meltshop."""
    effective_confirmation = confirmation or TargetConfirmation(
        status=ConfirmationStatus.UNCONFIRMED,
        notes=(
            "Awaiting Egyptian factory sign-off on hot-heel estimation, ladle-skull "
            "crediting, and weighbridge calibration semantics."
        ),
    )

    features: tuple[FeatureContract, ...] = (
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
            aliases=("heat_no", "heat_number", "melt_id", "رقم الصبة", "كود الصبة"),
            source_system="Level-2 MES",
        ),
        FeatureContract(
            name="production_order_id",
            display_name="Production Order Identifier",
            description=(
                "Campaign or customer production order grouping sequential heats; "
                "used for group-aware leakage-free splitting."
            ),
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("order_id", "work_order_id", "campaign_order_id", "امر الشغل"),
            source_system="ERP",
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
            aliases=("start_time", "heat_start_ts", "تاريخ الصبة", "وقت بداية الصبة"),
            source_system="MES Scheduler",
        ),
        FeatureContract(
            name="furnace_id",
            display_name="EAF Vessel Identifier",
            description="Physical electric arc furnace vessel asset identifier.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.STATIC,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("eaf_id", "furnace_no", "رقم الفرن"),
            source_system="MES Master Data",
        ),
        FeatureContract(
            name="shift_code",
            display_name="Planned Shift Code",
            description="Scheduled plant operating shift for the heat.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("shift", "shift_name", "الوردية"),
            constraints=FeatureConstraints(
                allowed_values=("MORNING", "EVENING", "NIGHT"),
            ),
            source_system="MES Calendar",
        ),
        FeatureContract(
            name="steel_grade_code",
            display_name="Target Steel Grade",
            description="Ordered steel metallurgy specification (e.g., B500DWR, B500B).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("steel_grade", "grade_code", "alloy_grade", "رتبة الصلب"),
            source_system="ERP / Metallurgy",
        ),
        FeatureContract(
            name="planned_scrap_mass_t",
            display_name="Planned Scrap Charge Mass",
            description="Planned total ferrous scrap mass loaded into charging baskets.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("scrap_mass_t", "planned_scrap_tons", "وزن الخردة المخطط"),
            constraints=FeatureConstraints(minimum=0.0, maximum=300.0),
            source_system="Scrap Yard Scale / MES",
        ),
        FeatureContract(
            name="planned_dri_mass_t",
            display_name="Planned DRI / HBI Charge Mass",
            description="Planned Direct Reduced Iron or HBI mass scheduled for the heat.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("dri_mass_t", "planned_dri_tons", "وزن الاختزال المباشر"),
            constraints=FeatureConstraints(minimum=0.0, maximum=300.0),
            source_system="DRI Silo Plan / MES",
        ),
        FeatureContract(
            name="planned_tap_mass_t",
            display_name="Planned Tap Mass",
            description="Target liquid steel tap weight ordered for the heat.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("target_tap_weight", "planned_tap_tons", "الوزن المستهدف للصبة"),
            constraints=FeatureConstraints(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=350.0,
            ),
            source_system="MES Heat Plan",
        ),
        FeatureContract(
            name="planned_scrap_grade_mix_code",
            display_name="Planned Scrap Grade Mix",
            description="Primary scrap grade classification for the scheduled charge.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("scrap_mix_code", "scrap_class", "تصنيف الخردة"),
            source_system="Scrap Yard Inventory",
        ),
        FeatureContract(
            name="expected_scrap_contaminant_pct",
            display_name="Expected Scrap Contaminant Share",
            description=(
                "Pre-charge expected non-metallic dirt, moisture, and debris percentage "
                "based on yard inspection or historical causal supplier prior (<= t0)."
            ),
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="%",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("prior_scrap_dirt_pct", "expected_dirt_pct", "نسبة الشوائب المتوقعة"),
            constraints=FeatureConstraints(minimum=0.0, maximum=25.0),
            source_system="Yard Inspection",
        ),
        FeatureContract(
            name="dri_metallization_pct",
            display_name="DRI Metallization Percentage",
            description="Pre-charge LIMS assay ratio of metallic Fe to total Fe in DRI lot.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.MEASURED,
            unit="%",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("metallization_pct", "dri_met_pct", "نسبة التمعدن"),
            constraints=FeatureConstraints(minimum=70.0, maximum=99.0),
            source_system="DRP LIMS",
        ),
        FeatureContract(
            name="dri_gangue_pct",
            display_name="DRI Acidic Gangue Percentage",
            description="Pre-charge LIMS assay non-metallic oxide (SiO2 + Al2O3) share in DRI.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.MEASURED,
            unit="%",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("gangue_pct", "شوائب الاختزال"),
            constraints=FeatureConstraints(minimum=0.0, maximum=15.0),
            source_system="DRP LIMS",
        ),
        FeatureContract(
            name="planned_hot_heel_mass_t",
            display_name="Planned Hot Heel Mass",
            description="Estimated liquid steel hot heel retained from the previous heat.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("hot_heel_t", "رصيد البوتقة الساخن"),
            constraints=FeatureConstraints(minimum=0.0, maximum=40.0),
            source_system="Meltshop Log",
        ),
        FeatureContract(
            name="planned_lime_flux_kg",
            display_name="Planned Lime and Dolomite Flux",
            description="Planned CaO / dolomite flux mass in the Level-2 melting recipe.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="kg",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("lime_kg", "flux_kg", "كمية الجير المخططة"),
            constraints=FeatureConstraints(minimum=0.0, maximum=20000.0),
            source_system="Level-2 Recipe",
        ),
        FeatureContract(
            name="planned_oxygen_nm3",
            display_name="Planned Oxygen Injection Volume",
            description="Planned oxygen lance and burner volume in the Level-2 melting recipe.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="Nm3",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("oxygen_plan_nm3", "كمية الاكسجين المخططة"),
            constraints=FeatureConstraints(minimum=0.0, maximum=15000.0),
            source_system="Level-2 Recipe",
        ),
        FeatureContract(
            name="target_tap_temp_c",
            display_name="Target Tapping Temperature",
            description="Ordered liquid steel tapping temperature specification.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="degC",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("aim_tap_temp", "حرارة الصب المستهدفة"),
            constraints=FeatureConstraints(minimum=1520.0, maximum=1720.0),
            source_system="Metallurgy Spec",
        ),
        FeatureContract(
            name="refractory_lining_heat_age",
            display_name="Refractory Lining Age (Heats)",
            description="Number of completed heats since the last EAF refractory relining.",
            data_type=DataType.INTEGER,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="heats",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("lining_age", "furnace_campaign_heat", "عمر التبطين الحراري"),
            constraints=FeatureConstraints(minimum=0.0, maximum=5000.0),
            source_system="Maintenance Log",
        ),
        # --- Deterministic derived pre-production features ---
        FeatureContract(
            name="planned_total_metallic_charge_t",
            display_name="Planned Total Metallic Charge",
            description="Deterministic sum of planned_scrap_mass_t and planned_dri_mass_t.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="metric_ton",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            constraints=FeatureConstraints(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=500.0,
            ),
            derived_from=("planned_scrap_mass_t", "planned_dri_mass_t"),
        ),
        FeatureContract(
            name="planned_dri_share_pct",
            display_name="Planned DRI Charge Share",
            description="Percentage of planned metallic charge supplied by DRI/HBI.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="%",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=100.0),
            derived_from=("planned_dri_mass_t", "planned_total_metallic_charge_t"),
        ),
        FeatureContract(
            name="planned_charge_to_tap_ratio",
            display_name="Planned Charge-to-Tap Ratio",
            description="Ratio of planned total metallic charge to planned liquid tap mass.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="ratio",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.5, maximum=2.0),
            derived_from=("planned_total_metallic_charge_t", "planned_tap_mass_t"),
        ),
        FeatureContract(
            name="planned_flux_kg_per_t_charge",
            display_name="Planned Flux Intensity",
            description="Planned lime/dolomite flux per metric ton of planned metallic charge.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="kg/t",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=500.0),
            derived_from=("planned_lime_flux_kg", "planned_total_metallic_charge_t"),
        ),
        FeatureContract(
            name="planned_oxygen_nm3_per_t_charge",
            display_name="Planned Oxygen Intensity",
            description="Planned oxygen volume per metric ton of planned metallic charge.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="Nm3/t",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=200.0),
            derived_from=("planned_oxygen_nm3", "planned_total_metallic_charge_t"),
        ),
        FeatureContract(
            name="planned_expected_dirt_mass_t",
            display_name="Planned Expected Scrap Dirt Mass",
            description="Expected non-metallic mass from scrap dirt/moisture before charging.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="metric_ton",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=100.0),
            derived_from=("planned_scrap_mass_t", "expected_scrap_contaminant_pct"),
        ),
    )

    target_feature = FeatureContract(
        name="metallic_waste_loss_t",
        display_name="Metallic Waste Loss",
        description=(
            "Measured metallic mass lost during EAF melting (FeO oxidation to slag, "
            "EAF baghouse dust, and non-recoverable metallic loss) per heat."
        ),
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit="metric_ton",
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("meltshop_waste_t", "yield_loss_t", "فاقد الصهر", "الهالك المعدني"),
        constraints=FeatureConstraints(minimum=0.0, maximum=100.0),
        source_system="Post-Tap Weighbridges",
    )

    target = TargetContract(
        target_id="eaf_metallic_waste_loss",
        version="1.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Net non-recovered metallic mass loss per EAF heat, reconciled from "
                "calibrated charge weighbridges and post-tap liquid steel ladle crane scale."
            ),
            granularity=EAF_GRANULARITY,
            measurement_method=(
                "Post-tap mass balance (actual_total_metallic_charge_t - "
                "actual_liquid_steel_tapped_t) after excluding credited recovered skulls."
            ),
            included_components=(
                "slag_iron_oxide_loss",
                "eaf_baghouse_dust_loss",
                "non_recoverable_splash_loss",
                "scrap_non_metallic_contaminants",
            ),
            excluded_components=(
                "returned_scrap_inventory",
                "credited_reusable_ladle_skull",
                "hot_heel_retained_in_vessel",
            ),
        ),
        confirmation=effective_confirmation,
        derived_rate=RateDefinition(
            name="metallic_waste_rate_pct",
            numerator="metallic_waste_loss_t",
            denominator="planned_total_metallic_charge_t",
            granularity=EAF_GRANULARITY,
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id=EAF_DATASET_ID,
        version="1.0.0",
        domain=DOMAIN_ID,
        process=EAF_PROCESS_ID,
        description=(
            "Pre-production EAF primary meltshop heat dataset for metallic waste loss prediction."
        ),
        granularity=EAF_GRANULARITY,
        features=features,
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description=(
                "Prediction executed before scrap basket charging and DRI feeding (t <= t0)."
            ),
        ),
        business_keys=("heat_id",),
        grouping_columns=("production_order_id",),
        time_column="planned_start_ts",
    )


def build_eaf_meltshop_process(
    *,
    confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the ProcessContract for EAF Primary Meltshop."""
    dataset_contract = build_eaf_meltshop_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=EAF_TASK_ID,
        version="1.0.0",
        display_name="EAF Pre-Production Metallic Waste Prediction",
        description=(
            "Predict expected metallic waste loss (metric tons) for an EAF heat "
            "using planned charge mix, DRI assays, recipe setpoints, and vessel age."
        ),
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    return ProcessContract(
        process_id=EAF_PROCESS_ID,
        version="1.0.0",
        display_name="Electric Arc Furnace (EAF) Primary Meltshop",
        description=(
            "Primary steelmaking melting of scrap and Direct Reduced Iron (DRI/HBI) "
            "into liquid steel prior to secondary metallurgy and continuous casting."
        ),
        tasks=(task,),
    )


def validate_eaf_physical_rules(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check EAF-specific physical consistency rules without altering raw values."""
    issues: list[ValidationIssue] = []

    if "planned_scrap_mass_t" in df.columns and "planned_dri_mass_t" in df.columns:
        scrap = pd.to_numeric(df["planned_scrap_mass_t"], errors="coerce")
        dri = pd.to_numeric(df["planned_dri_mass_t"], errors="coerce")
        total = scrap.fillna(0.0) + dri.fillna(0.0)
        bad_mask = total <= 0.0
        if bool(bad_mask.any()):
            bad_rows = tuple(int(i) for i in np.flatnonzero(bad_mask)[:20])
            issues.append(
                ValidationIssue(
                    code="EAF_ZERO_TOTAL_CHARGE",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Combined planned scrap + DRI charge mass must be strictly positive."
                    ),
                    columns=("planned_scrap_mass_t", "planned_dri_mass_t"),
                    affected_rows=int(bad_mask.sum()),
                    row_samples=bad_rows,
                    recommendation="Verify planned_scrap_mass_t and planned_dri_mass_t entries.",
                )
            )

    if (
        "metallic_waste_loss_t" in df.columns
        and "planned_scrap_mass_t" in df.columns
        and "planned_dri_mass_t" in df.columns
    ):
        waste = pd.to_numeric(df["metallic_waste_loss_t"], errors="coerce")
        total_plan = pd.to_numeric(df["planned_scrap_mass_t"], errors="coerce").fillna(
            0.0
        ) + pd.to_numeric(df["planned_dri_mass_t"], errors="coerce").fillna(0.0)
        impossible = waste.notna() & (total_plan > 0) & (waste > total_plan * 1.15)
        if bool(impossible.any()):
            bad_rows = tuple(int(i) for i in np.flatnonzero(impossible)[:20])
            issues.append(
                ValidationIssue(
                    code="EAF_WASTE_EXCEEDS_CHARGE",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Measured metallic waste loss exceeds 115% of planned metallic charge; "
                        "check unit mismatch (kg vs metric_ton) or weighbridge error."
                    ),
                    columns=(
                        "metallic_waste_loss_t",
                        "planned_scrap_mass_t",
                        "planned_dri_mass_t",
                    ),
                    affected_rows=int(impossible.sum()),
                    row_samples=bad_rows,
                    recommendation="Confirm mass units and reconcile weighbridge records.",
                )
            )

    return issues


def engineer_eaf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute verified deterministic pre-production EAF features row-by-row."""
    out = df.copy()
    scrap = pd.to_numeric(out["planned_scrap_mass_t"], errors="coerce")
    dri = pd.to_numeric(out["planned_dri_mass_t"], errors="coerce")
    total_charge = scrap + dri
    out["planned_total_metallic_charge_t"] = total_charge

    safe_charge = total_charge.where(total_charge > 0.0, np.nan)
    out["planned_dri_share_pct"] = (dri / safe_charge) * 100.0

    tap = pd.to_numeric(out["planned_tap_mass_t"], errors="coerce")
    safe_tap = tap.where(tap > 0.0, np.nan)
    out["planned_charge_to_tap_ratio"] = total_charge / safe_tap

    if "planned_lime_flux_kg" in out.columns:
        flux = pd.to_numeric(out["planned_lime_flux_kg"], errors="coerce")
        out["planned_flux_kg_per_t_charge"] = flux / safe_charge
    else:
        out["planned_flux_kg_per_t_charge"] = np.nan

    if "planned_oxygen_nm3" in out.columns:
        oxygen = pd.to_numeric(out["planned_oxygen_nm3"], errors="coerce")
        out["planned_oxygen_nm3_per_t_charge"] = oxygen / safe_charge
    else:
        out["planned_oxygen_nm3_per_t_charge"] = np.nan

    if "expected_scrap_contaminant_pct" in out.columns:
        contam_pct = pd.to_numeric(out["expected_scrap_contaminant_pct"], errors="coerce")
        out["planned_expected_dirt_mass_t"] = scrap * (contam_pct / 100.0)
    else:
        out["planned_expected_dirt_mass_t"] = np.nan

    return out
