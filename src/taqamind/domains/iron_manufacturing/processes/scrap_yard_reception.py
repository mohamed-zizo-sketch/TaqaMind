"""Scrap Yard Reception process contract, physical rules, and feature engineering.

Industrial scope:
    Ferrous scrap truck/rail reception and non-metallic contaminant inspection
    (soil, water, other non-Fe debris) prior to furnace charging.
    Backed by real EU Horizon ALCHIMIA datasets (`scrap_A.csv`, `scrap_C.csv`, CC-BY-4.0)
    for EDA and reference benchmarking — NOT claimed as Egyptian validation.
Row granularity:
    One incoming scrap delivery vehicle (`scrap_delivery_id`).
Prediction timing:
    Pre-inspection (`t <= t0`) at entry weighbridge, before unloading inspection
    and exit weighing (`ExitDate` and `Difference` are strictly excluded as leakage).
Target:
    `total_contaminant_kg` (kg) = `soil_kg + water_kg + other_kg`.
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

SCRAP_YARD_PROCESS_ID = "scrap_yard_reception"
SCRAP_YARD_TASK_ID = "scrap_delivery_contaminant_prediction"
SCRAP_YARD_DATASET_ID = "scrap_yard_delivery_contaminant"
SCRAP_YARD_GRANULARITY = "one_scrap_delivery_vehicle"

SCRAP_YARD_FORBIDDEN_PREDICTOR_NAMES: frozenset[str] = frozenset(
    {
        "exit_ts",
        "exitdate",
        "stay_duration_min",
        "difference",
        "soil_kg",
        "water_kg",
        "other_kg",
        "soil",
        "water",
        "other",
    }
)


def build_scrap_yard_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the authoritative Shared-Core DatasetContract for Scrap Yard Reception."""
    effective_confirmation = confirmation or TargetConfirmation(
        status=ConfirmationStatus.UNCONFIRMED,
        notes=(
            "Awaiting factory confirmation on whether Soil/Water/Other contaminant "
            "deductions are physically weighed or inspector-estimated and whether "
            "gross_delivery_mass_kg is recorded at entry weighbridge before unloading."
        ),
    )

    features: tuple[FeatureContract, ...] = (
        FeatureContract(
            name="scrap_delivery_id",
            display_name="Scrap Delivery Ticket ID",
            description="Unique business identifier for one incoming scrap vehicle delivery.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("delivery_id", "ticket_id", "weighbridge_id", "رقم الشحنة", "اذن الاستلام"),
            source_system="Weighbridge System",
        ),
        FeatureContract(
            name="supplier_group_id",
            display_name="Supplier Group Key",
            description=(
                "Supplier identifier used for group-aware splitting and leakage control "
                "(prevents memorizing single-supplier bursts across train/test)."
            ),
            data_type=DataType.STRING,
            role=FeatureRole.METADATA,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("vendor_group_id", "supplier_batch_group"),
            source_system="ERP Procurement",
        ),
        FeatureContract(
            name="entry_ts",
            display_name="Weighbridge Entry Timestamp",
            description="Timestamp when the scrap vehicle arrives at the entry weighbridge (t0).",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("entrydate", "entry_date", "planned_start_ts", "تاريخ الدخول"),
            source_system="Weighbridge System",
        ),
        FeatureContract(
            name="supplier_code",
            display_name="Scrap Supplier Code",
            description="Categorical code identifying the supplying scrap yard or broker.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("supplier", "supplier_id", "vendor_id", "المورد"),
            source_system="ERP Procurement",
        ),
        FeatureContract(
            name="scrap_grade_code",
            display_name="Declared Scrap Grade",
            description="Commercial scrap grade code declared on the delivery manifest.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("scraptype", "scrap_type", "درجة الخردة", "صنف الخردة"),
            source_system="Weighbridge Manifest",
        ),
        FeatureContract(
            name="gross_delivery_mass_kg",
            display_name="Delivery Scrap Mass",
            description="Weighed delivery mass of the incoming scrap load (kg).",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.MEASURED,
            unit="kg",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("delivery_weight_kg", "truck_weight_kg", "وزن الشحنة"),
            constraints=FeatureConstraints(
                minimum=0.0,
                minimum_inclusive=False,
                maximum=200_000.0,
            ),
            source_system="Weighbridge Scale",
        ),
        FeatureContract(
            name="warehouse_zone",
            display_name="Assigned Yard Warehouse Zone",
            description="Designated scrap yard bay or warehouse zone assigned at entry.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("warehouse", "yard_zone", "المخزن"),
            source_system="Yard Dispatch",
        ),
        FeatureContract(
            name="transport_mode",
            display_name="Delivery Transport Mode",
            description="Transport modality delivering the scrap load (e.g., Truck, Rail car).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            aliases=("transport", "vehicle_type", "وسيلة النقل"),
            source_system="Gate Log",
        ),
        # --- Deterministic derived pre-production features ---
        FeatureContract(
            name="delivery_mass_metric_ton",
            display_name="Delivery Mass (metric tons)",
            description="Exact SI conversion of gross_delivery_mass_kg / 1000.0.",
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
                maximum=200.0,
            ),
            derived_from=("gross_delivery_mass_kg",),
        ),
        FeatureContract(
            name="entry_hour_of_day",
            display_name="Entry Hour of Day",
            description="Deterministic arrival hour (0-23) extracted from entry_ts.",
            data_type=DataType.INTEGER,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.DERIVED,
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=0.0, maximum=23.0),
            derived_from=("entry_ts",),
        ),
    )

    target_feature = FeatureContract(
        name="total_contaminant_kg",
        display_name="Total Non-Metallic Contaminant Mass",
        description=(
            "Measured or deducted non-metallic contaminant mass (Soil + Water + Other) "
            "for one scrap delivery in kilograms."
        ),
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit="kg",
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("contaminant_kg", "اجمالي الشوائب"),
        constraints=FeatureConstraints(minimum=0.0, maximum=200_000.0),
        source_system="Scrap Yard Inspection",
    )

    target = TargetContract(
        target_id="scrap_delivery_total_contaminant",
        version="1.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Total non-metallic waste mass (soil, water/moisture, and other debris) "
                "deducted upon unloading inspection of a scrap delivery."
            ),
            granularity=SCRAP_YARD_GRANULARITY,
            measurement_method="Sum of soil_kg + water_kg + other_kg logged at unloading.",
            included_components=(
                "soil_dirt_deduction",
                "water_moisture_deduction",
                "other_non_metallic_debris",
            ),
            excluded_components=(
                "recoverable_ferrous_scrap_mass",
                "rejected_full_truck_returned_to_supplier",
            ),
        ),
        confirmation=effective_confirmation,
        derived_rate=RateDefinition(
            name="contaminant_rate_pct",
            numerator="total_contaminant_kg",
            denominator="gross_delivery_mass_kg",
            granularity=SCRAP_YARD_GRANULARITY,
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id=SCRAP_YARD_DATASET_ID,
        version="1.0.0",
        domain=DOMAIN_ID,
        process=SCRAP_YARD_PROCESS_ID,
        description=(
            "Pre-inspection scrap delivery dataset for non-metallic contaminant prediction."
        ),
        granularity=SCRAP_YARD_GRANULARITY,
        features=features,
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description=(
                "Prediction executed at entry weighbridge before unloading inspection (t <= t0)."
            ),
        ),
        business_keys=("scrap_delivery_id",),
        grouping_columns=("supplier_group_id",),
        time_column="entry_ts",
    )


def build_scrap_yard_process(
    *,
    confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the ProcessContract for Scrap Yard Reception."""
    dataset_contract = build_scrap_yard_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=SCRAP_YARD_TASK_ID,
        version="1.0.0",
        display_name="Scrap Yard Pre-Inspection Contaminant Prediction",
        description=(
            "Predict expected non-metallic contaminant mass (kg) in an incoming scrap "
            "delivery from supplier, scrap grade, delivery mass, transport mode, and arrival time."
        ),
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    return ProcessContract(
        process_id=SCRAP_YARD_PROCESS_ID,
        version="1.0.0",
        display_name="Scrap Yard Reception & Quality Inspection",
        description=(
            "Incoming ferrous scrap weighbridge reception and non-metallic contaminant "
            "assessment prior to EAF charging."
        ),
        tasks=(task,),
    )


def validate_scrap_yard_physical_rules(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check physical bounds on scrap delivery mass vs contaminant deduction."""
    issues: list[ValidationIssue] = []
    if "total_contaminant_kg" in df.columns and "gross_delivery_mass_kg" in df.columns:
        contam = pd.to_numeric(df["total_contaminant_kg"], errors="coerce")
        gross = pd.to_numeric(df["gross_delivery_mass_kg"], errors="coerce")
        impossible = contam.notna() & gross.notna() & (gross > 0) & (contam > gross)
        if bool(impossible.any()):
            bad_rows = tuple(int(i) for i in np.flatnonzero(impossible)[:20])
            issues.append(
                ValidationIssue(
                    code="SCRAP_CONTAMINANT_EXCEEDS_DELIVERY_MASS",
                    stage=PipelineStage.SEMANTIC_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message="Total contaminant mass (kg) exceeds gross delivery mass (kg).",
                    columns=("total_contaminant_kg", "gross_delivery_mass_kg"),
                    affected_rows=int(impossible.sum()),
                    row_samples=bad_rows,
                    recommendation="Verify delivery weighbridge mass and contaminant units.",
                )
            )
    return issues


def engineer_scrap_yard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute verified deterministic pre-inspection scrap yard features."""
    out = df.copy()
    gross_kg = pd.to_numeric(out["gross_delivery_mass_kg"], errors="coerce")
    out["delivery_mass_metric_ton"] = gross_kg / 1000.0
    parsed_ts = pd.to_datetime(out["entry_ts"], errors="coerce")
    out["entry_hour_of_day"] = parsed_ts.dt.hour.astype("float64")
    return out
