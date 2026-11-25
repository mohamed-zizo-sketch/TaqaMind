"""Rebar Hot Rolling Mill process contract, physical rules, and feature engineering.

Industrial scope:
    Egyptian billet-to-rebar hot rolling mills (ES 262-2 / B500DWR long products).
Row granularity:
    One rolling campaign lot (`rolling_lot_id`).
Prediction timing:
    Strictly pre-production (`t <= t0`), when billet batch is scheduled prior to
    reheating furnace entry.
Target:
    `total_rolling_waste_kg` (kg) — sum of crop ends, cobble scrap, mill scale,
    and quality reject bars. Unconfirmed until factory scale methodology is signed off.
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

REBAR_PROCESS_ID = "rebar_hot_rolling"
REBAR_TASK_ID = "rebar_rolling_waste_quantity_prediction"
REBAR_DATASET_ID = "rebar_rolling_lot_waste"
REBAR_GRANULARITY = "one_rolling_campaign_lot"

REBAR_FORBIDDEN_PREDICTOR_NAMES: frozenset[str] = frozenset(
    {
        "crop_ends_kg",
        "cobble_scrap_kg",
        "mill_scale_kg",
        "quality_reject_bars_kg",
        "actual_good_rebar_output_t",
        "reheat_gas_nm3",
        "mill_electricity_kwh",
        "actual_discharge_temp_c",
        "actual_cobble_downtime_min",
        "actual_power_factor",
        "rolling_yield_pct",
    }
)


def build_rebar_rolling_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the authoritative Shared-Core DatasetContract for Rebar Hot Rolling."""
    effective_confirmation = confirmation or TargetConfirmation(
        status=ConfirmationStatus.UNCONFIRMED,
        notes=(
            "Awaiting Egyptian rolling mill sign-off on mill-scale estimation vs "
            "direct weighing and recoverable crop-end classification."
        ),
    )

    features: tuple[FeatureContract, ...] = (
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
            aliases=("lot_id", "campaign_id", "rolling_batch_id", "رقم التشغيلة", "رقم اللوط"),
            source_system="Rolling MES",
        ),
        FeatureContract(
            name="production_order_id",
            display_name="Production Order Identifier",
            description="Customer or campaign ERP order grouping multiple rolling lots.",
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("order_id", "work_order_id", "امر الشغل"),
            source_system="ERP",
        ),
        FeatureContract(
            name="planned_start_ts",
            display_name="Planned Lot Start Timestamp",
            description="Scheduled timestamp when billet lot enters the reheating furnace (t0).",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("start_time", "lot_date", "تاريخ التشغيل"),
            source_system="Rolling MES",
        ),
        FeatureContract(
            name="mill_line_id",
            display_name="Rolling Mill Line Identifier",
            description="Physical hot rolling mill line asset identifier.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.STATIC,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("mill_id", "line_id", "خط الدرفلة"),
            source_system="MES Master Data",
        ),
        FeatureContract(
            name="steel_grade_code",
            display_name="Billet Steel Grade",
            description="Steel grade specification of the charged billets (e.g., B500DWR).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("steel_grade", "grade_code", "رتبة الصلب"),
            source_system="ERP / Metallurgy",
        ),
        FeatureContract(
            name="rebar_diameter_mm",
            display_name="Target Rebar Diameter",
            description="Ordered finished rebar nominal diameter (mm).",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="mm",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("bar_diameter_mm", "diameter_mm", "قطر السيخ"),
            constraints=FeatureConstraints(minimum=6.0, maximum=50.0),
            source_system="Rolling Order Spec",
        ),
        FeatureContract(
            name="planned_input_billet_mass_t",
            display_name="Planned Input Billet Mass",
            description="Total planned billet steel mass charged to the rolling lot (metric tons).",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("billet_input_mass_t", "input_billet_tons", "وزن البيلت الداخل"),
            constraints=FeatureConstraints(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=5000.0,
            ),
            source_system="Billet Yard Scale",
        ),
        FeatureContract(
            name="slitting_strands_count",
            display_name="Slitting Strands Count",
            description="Number of parallel slit rebar strands configured in the pass design.",
            data_type=DataType.INTEGER,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="strands",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("slit_strands", "strands_count", "عدد خطوط الشق"),
            constraints=FeatureConstraints(minimum=1.0, maximum=6.0),
            source_system="Mill Pass Design",
        ),
        FeatureContract(
            name="planned_billet_count",
            display_name="Planned Billet Count",
            description="Scheduled number of individual steel billets in the rolling lot.",
            data_type=DataType.INTEGER,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="billets",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("billet_count", "عدد البيلت"),
            constraints=FeatureConstraints(minimum=1.0, maximum=10000.0),
            source_system="Billet Yard Scale",
        ),
        FeatureContract(
            name="charging_mode_hot_or_cold",
            display_name="Billet Charging Mode",
            description="Whether billets enter directly hot from CCM or cold from yard.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("charging_mode", "hot_cold_charge", "نوع الشحن"),
            constraints=FeatureConstraints(allowed_values=("HOT_CHARGE", "COLD_CHARGE")),
            source_system="Mill Scheduler",
        ),
        FeatureContract(
            name="target_reheat_temp_c",
            display_name="Target Reheat Furnace Temperature",
            description="Planned billet discharge temperature setpoint at the reheating furnace.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="degC",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("aim_reheat_temp", "حرارة التسخين المستهدفة"),
            constraints=FeatureConstraints(minimum=950.0, maximum=1300.0),
            source_system="Furnace Recipe",
        ),
        FeatureContract(
            name="finishing_roll_groove_tonnage_age_t",
            display_name="Finishing Roll Groove Tonnage Age",
            description="Cumulative metric tons rolled on the current finishing stand roll groove.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="metric_ton",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("groove_age_t", "roll_tonnage_age", "عمر الدرفيل"),
            constraints=FeatureConstraints(minimum=0.0, maximum=50000.0),
            source_system="Roll Shop Log",
        ),
        FeatureContract(
            name="planned_mill_speed_mps",
            display_name="Planned Finishing Mill Speed",
            description="Scheduled finishing stand rolling velocity (m/s).",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="m/s",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("mill_speed_mps", "سرعة الدرفلة"),
            constraints=FeatureConstraints(minimum=1.0, maximum=60.0),
            source_system="Mill PLC Recipe",
        ),
        # --- Deterministic derived pre-production features ---
        FeatureContract(
            name="planned_input_billet_mass_kg",
            display_name="Planned Input Billet Mass (kg)",
            description=(
                "Exact SI conversion of planned_input_billet_mass_t (* 1000) into kg "
                "for unit-compatible waste rate calculations."
            ),
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="kg",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            constraints=FeatureConstraints(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=5_000_000.0,
            ),
            derived_from=("planned_input_billet_mass_t",),
        ),
        FeatureContract(
            name="planned_single_billet_mass_t",
            display_name="Planned Single Billet Mass",
            description="Average planned billet mass = planned_input_billet_mass_t / billet count.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            unit="metric_ton",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.05, maximum=20.0),
            derived_from=("planned_input_billet_mass_t", "planned_billet_count"),
        ),
        FeatureContract(
            name="is_multi_slit_flag",
            display_name="Multi-Slit Rolling Indicator",
            description="Binary indicator (1 if slitting_strands_count >= 2 else 0).",
            data_type=DataType.INTEGER,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=1.0),
            derived_from=("slitting_strands_count",),
        ),
    )

    target_feature = FeatureContract(
        name="total_rolling_waste_kg",
        display_name="Total Rolling Waste Quantity",
        description=(
            "Measured total rolling scrap and scale loss per rolling lot "
            "(crop ends + cobble scrap + mill scale + quality rejects) in kg."
        ),
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit="kg",
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("rolling_waste_kg", "اجمالي هالك الدرفلة"),
        constraints=FeatureConstraints(minimum=0.0, maximum=1_000_000.0),
        source_system="Post-Roll Scrap Scales",
    )

    target = TargetContract(
        target_id="rebar_total_rolling_waste",
        version="1.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Total measured rolling material loss per campaign lot combining crop ends, "
                "cobble scrap, mill scale oxidation, and quality-rejected rebar."
            ),
            granularity=REBAR_GRANULARITY,
            measurement_method=(
                "Sum of weighed post-rolling scrap bins (crop_ends_kg + cobble_scrap_kg + "
                "quality_reject_bars_kg) plus reconciled mill_scale_kg."
            ),
            included_components=(
                "crop_ends_scrap",
                "cobble_jam_scrap",
                "mill_scale_oxidation_loss",
                "dimensional_quality_reject_bars",
            ),
            excluded_components=(
                "unrolled_billets_returned_to_yard",
                "reclassified_prime_short_bars_sold",
            ),
        ),
        confirmation=effective_confirmation,
        derived_rate=RateDefinition(
            name="rolling_waste_rate_pct",
            numerator="total_rolling_waste_kg",
            denominator="planned_input_billet_mass_kg",
            granularity=REBAR_GRANULARITY,
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id=REBAR_DATASET_ID,
        version="1.0.0",
        domain=DOMAIN_ID,
        process=REBAR_PROCESS_ID,
        description=(
            "Pre-production rebar hot rolling lot dataset for total rolling waste prediction."
        ),
        granularity=REBAR_GRANULARITY,
        features=features,
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description=(
                "Prediction executed before billet lot enters the reheating furnace (t <= t0)."
            ),
        ),
        business_keys=("rolling_lot_id",),
        grouping_columns=("production_order_id",),
        time_column="planned_start_ts",
    )


def build_rebar_hot_rolling_process(
    *,
    confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the ProcessContract for Rebar Hot Rolling Mill."""
    dataset_contract = build_rebar_rolling_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=REBAR_TASK_ID,
        version="1.0.0",
        display_name="Rebar Hot Rolling Pre-Production Waste Prediction",
        description=(
            "Predict expected total rolling waste quantity (kg) for a billet rolling lot "
            "from diameter, slitting strands, billet mass, charging mode, and roll groove age."
        ),
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    return ProcessContract(
        process_id=REBAR_PROCESS_ID,
        version="1.0.0",
        display_name="Rebar Hot Rolling Mill",
        description=(
            "Hot rolling of continuous-cast steel billets through reheating furnace, "
            "roughing/intermediate/slitting stands, and cooling bed into finished rebar."
        ),
        tasks=(task,),
    )


def validate_rebar_physical_rules(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check rebar rolling physical balance rules."""
    issues: list[ValidationIssue] = []
    if "total_rolling_waste_kg" in df.columns and "planned_input_billet_mass_t" in df.columns:
        waste_kg = pd.to_numeric(df["total_rolling_waste_kg"], errors="coerce")
        input_kg = pd.to_numeric(df["planned_input_billet_mass_t"], errors="coerce") * 1000.0
        impossible = waste_kg.notna() & input_kg.notna() & (input_kg > 0) & (waste_kg > input_kg)
        if bool(impossible.any()):
            bad_rows = tuple(int(i) for i in np.flatnonzero(impossible)[:20])
            issues.append(
                ValidationIssue(
                    code="REBAR_WASTE_EXCEEDS_INPUT_MASS",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(
                        "Total rolling waste (kg) exceeds 100% of planned input billet mass (kg)."
                    ),
                    columns=("total_rolling_waste_kg", "planned_input_billet_mass_t"),
                    affected_rows=int(impossible.sum()),
                    row_samples=bad_rows,
                    recommendation="Verify unit alignment (kg vs metric_ton) on billet and waste.",
                )
            )
    return issues


def engineer_rebar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute verified deterministic pre-production rebar rolling features."""
    out = df.copy()
    billet_t = pd.to_numeric(out["planned_input_billet_mass_t"], errors="coerce")
    out["planned_input_billet_mass_kg"] = billet_t * 1000.0

    if "planned_billet_count" in out.columns:
        count = pd.to_numeric(out["planned_billet_count"], errors="coerce")
        safe_count = count.where(count > 0.0, np.nan)
        out["planned_single_billet_mass_t"] = billet_t / safe_count
    else:
        out["planned_single_billet_mass_t"] = np.nan

    if "slitting_strands_count" in out.columns:
        strands = pd.to_numeric(out["slitting_strands_count"], errors="coerce")
        out["is_multi_slit_flag"] = np.where(strands.notna(), (strands >= 2).astype(float), np.nan)
    else:
        out["is_multi_slit_flag"] = np.nan

    return out
