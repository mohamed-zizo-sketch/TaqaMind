"""Knitting V1 contracts for pre-production waste quantity regression."""

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
    ProcessContract,
    RateDefinition,
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)
from taqamind.core.feature_engineering import DerivedFeatureSpec
from taqamind.domains.textile.common.formulas import compute_knitting_tightness_factor

KNITTING_PROCESS_ID = "knitting"
KNITTING_TASK_ID = "knitting_waste_regression"
KNITTING_DATASET_ID = "knitting_shift_runs"

KNITTING_POST_PRODUCTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "Shift Total",
    "Machine Total",
    "SHIFT-A",
    "SHIFT-B",
    "SHIFT-C",
    "Without Shift(Outbound)",
)


def build_synthetic_knitting_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for test fixtures and offline demos."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="synthetic_textile_test_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic fixture confirmation only; not a real factory sign-off.",
    )


def build_knitting_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Knitting V1 DatasetContract (UNCONFIRMED by default)."""
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
        aliases=("Lot",),
    )
    production_date = FeatureContract(
        name="production_date",
        display_name="Scheduled Production Date",
        description="Scheduled calendar date of the circular knitting run.",
        data_type=DataType.DATE,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Production Date",),
    )
    planned_qty_kg = FeatureContract(
        name="planned_qty_kg",
        display_name="Planned Production Quantity (kg)",
        description="Planned yarn mass allocated for the knitting shift run.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
    )
    machine_brand = FeatureContract(
        name="machine_brand",
        display_name="Machine Brand",
        description="Circular knitting machine manufacturer.",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        aliases=("M/C Brand",),
    )
    machine_diameter_in = FeatureContract(
        name="machine_diameter_in",
        display_name="Machine Cylinder Diameter (in)",
        description="Circular knitting cylinder diameter in inches.",
        data_type=DataType.FLOAT,
        unit="in",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("m/c_dia",),
    )
    machine_gauge_e = FeatureContract(
        name="machine_gauge_e",
        display_name="Machine Gauge (E)",
        description="Needles per inch (machine gauge).",
        data_type=DataType.FLOAT,
        unit="E",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("m/c_gauge",),
    )
    yarn_count_tex_1 = FeatureContract(
        name="yarn_count_tex_1",
        display_name="Primary Yarn Linear Density (tex)",
        description="Primary feeder yarn linear density in tex.",
        data_type=DataType.FLOAT,
        unit="tex",
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
        aliases=("stitch_1", "stitch"),
    )
    target_gsm = FeatureContract(
        name="target_gsm",
        display_name="Target Fabric GSM (g/m2)",
        description="Specified target fabric areal weight in grams per square meter.",
        data_type=DataType.FLOAT,
        unit="g/m2",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=20.0),
        aliases=("GSM",),
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
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=1),
    )
    knitting_tightness_factor = FeatureContract(
        name="knitting_tightness_factor",
        display_name="Knitting Tightness Factor (K)",
        description="Derived Munden tightness factor sqrt(tex) / (stitch_length_mm / 10).",
        data_type=DataType.FLOAT,
        unit="tex^0.5/cm",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=0.1),
        derived_from=("yarn_count_tex_1", "stitch_length_mm"),
    )

    waste_qty_kg = FeatureContract(
        name="waste_qty_kg",
        display_name="Knitting Waste Quantity (kg)",
        description="Measured rejected/scrap knitted fabric mass per shift run.",
        data_type=DataType.FLOAT,
        unit="kg",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Reject Qty",),
    )

    target = TargetContract(
        target_id=KNITTING_TASK_ID,
        feature=waste_qty_kg,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured knitted fabric waste/rejection mass (kg) per scheduled shift run."
            ),
            granularity="shift_run",
            measurement_method="Post-shift digital scale weighing at circular knitting machine.",
            included_components=("running_defect_scrap", "setup_changeover_scrap"),
            excluded_components=("good_fabric_rolls", "unlogged_outbound"),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="knitting_waste_rate_pct",
            numerator="waste_qty_kg",
            denominator="planned_qty_kg",
            granularity="shift_run",
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id=KNITTING_DATASET_ID,
        version="1.0.0",
        domain="textile",
        process=KNITTING_PROCESS_ID,
        description=(
            "Pre-production circular knitting waste quantity prediction contract at "
            "shift-run granularity with lot grouping and scheduled date."
        ),
        granularity="shift_run",
        features=(
            run_id,
            order_lot_id,
            production_date,
            planned_qty_kg,
            machine_brand,
            machine_diameter_in,
            machine_gauge_e,
            yarn_count_tex_1,
            stitch_length_mm,
            target_gsm,
            number_of_feeders,
            knitting_tightness_factor,
        ),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict knitting scrap mass (kg) before starting machine shift run.",
        ),
        business_keys=("run_id",),
        grouping_columns=("order_lot_id",),
        time_column="production_date",
    )


def build_knitting_derived_specs() -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs for knitting."""
    return (
        DerivedFeatureSpec(
            feature_name="knitting_tightness_factor",
            compute=lambda df: compute_knitting_tightness_factor(
                df["yarn_count_tex_1"], df["stitch_length_mm"]
            ),
        ),
    )


def build_knitting_process_contract(
    *,
    confirmation: TargetConfirmation | None = None,
    energy_confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the Knitting ProcessContract for DomainRegistry registration."""
    from taqamind.domains.textile.knitting.energy_contracts import (
        build_knitting_energy_task_contract,
    )

    dataset_contract = build_knitting_dataset_contract(confirmation=confirmation)
    waste_task = IndustrialTaskContract(
        task_id=KNITTING_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Knitting Waste Quantity Regression",
        description="Forecast expected circular knitting waste mass (kg) per shift run.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    energy_task = build_knitting_energy_task_contract(confirmation=energy_confirmation)
    return ProcessContract(
        process_id=KNITTING_PROCESS_ID,
        version="1.0.0",
        display_name="Knitting",
        description="Circular and flat weft knitting production process.",
        tasks=(waste_task, energy_task),
    )
