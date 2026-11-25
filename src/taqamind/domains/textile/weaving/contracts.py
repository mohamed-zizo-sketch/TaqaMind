"""Weaving V1 contracts for pre-production waste length regression."""

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
from taqamind.domains.textile.common.formulas import compute_weaving_thread_density_sum

WEAVING_PROCESS_ID = "weaving"
WEAVING_TASK_ID = "weaving_waste_regression"
WEAVING_DATASET_ID = "weaving_beam_batches"

# Post-production fields present in historical weaving logs that are strictly prohibited
# as pre-production predictors.
WEAVING_POST_PRODUCTION_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "act_shrink%",
    "Total_pdn_m/c",
    "Total_pdn_per_order",
    "Previous_pdn",
)


def build_synthetic_weaving_confirmation() -> TargetConfirmation:
    """Explicit synthetic target confirmation strictly for test fixtures and offline demos."""
    return TargetConfirmation(
        status=ConfirmationStatus.CONFIRMED,
        confirmed_by="synthetic_textile_test_fixture",
        confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        notes="Synthetic fixture confirmation only; not a real factory sign-off.",
    )


def build_weaving_dataset_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the Weaving V1 DatasetContract (UNCONFIRMED by default)."""
    batch_record_id = FeatureContract(
        name="batch_record_id",
        display_name="Beam Batch Record ID",
        description="Unique weaving beam batch or loom-run record identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Record_ID",),
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
        aliases=("ID",),
    )
    planned_beam_length_yds = FeatureContract(
        name="planned_beam_length_yds",
        display_name="Planned Beam Length (yds)",
        description="Planned weaver beam length prior to loom execution.",
        data_type=DataType.FLOAT,
        unit="yd",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Req_beam_length(yds)", "Rec_Beam_length(yds)"),
    )
    planned_finish_fabric_yds = FeatureContract(
        name="planned_finish_fabric_yds",
        display_name="Planned Finished Fabric (yds)",
        description="Target finished fabric length planned for the weaving order.",
        data_type=DataType.FLOAT,
        unit="yd",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Req_Finish_Fabrics",),
    )
    planned_grey_fabric_yds = FeatureContract(
        name="planned_grey_fabric_yds",
        display_name="Planned Grey Fabric (yds)",
        description="Planned grey fabric yardage before finishing.",
        data_type=DataType.FLOAT,
        unit="yd",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Req_grey_fabric",),
    )
    fabric_allowance_pct = FeatureContract(
        name="fabric_allowance_pct",
        display_name="Fabric Allowance (%)",
        description="Planned finishing and process fabric allowance percentage.",
        data_type=DataType.FLOAT,
        unit="%",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=0.0),
        aliases=("Fabric_Allowance",),
    )
    shrink_allowance_pct = FeatureContract(
        name="shrink_allowance_pct",
        display_name="Shrinkage Allowance (%)",
        description="Expected pre-production shrinkage allowance percentage.",
        data_type=DataType.FLOAT,
        unit="%",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=0.0),
        aliases=("Shrink_allow",),
    )
    warp_count_ne = FeatureContract(
        name="warp_count_ne",
        display_name="Warp Yarn Count (Ne)",
        description="English cotton count of longitudinal warp yarn.",
        data_type=DataType.FLOAT,
        unit="Ne",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("warp_count",),
    )
    weft_count_ne = FeatureContract(
        name="weft_count_ne",
        display_name="Weft Yarn Count (Ne)",
        description="English cotton count of transverse weft yarn.",
        data_type=DataType.FLOAT,
        unit="Ne",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("weft_count",),
    )
    warp_ends_per_inch = FeatureContract(
        name="warp_ends_per_inch",
        display_name="Warp Ends Per Inch (EPI)",
        description="Number of warp threads per inch in fabric specification.",
        data_type=DataType.FLOAT,
        unit="ends/in",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("epi",),
    )
    weft_picks_per_inch = FeatureContract(
        name="weft_picks_per_inch",
        display_name="Weft Picks Per Inch (PPI)",
        description="Number of weft picks per inch in fabric specification.",
        data_type=DataType.FLOAT,
        unit="picks/in",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        constraints=FeatureConstraints(minimum=1.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("ppi",),
    )
    fabric_construction = FeatureContract(
        name="fabric_construction",
        display_name="Fabric Construction",
        description="Planned woven fabric construction code.",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.CONFIGURED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        aliases=("Construction",),
    )
    total_thread_density_per_inch = FeatureContract(
        name="total_thread_density_per_inch",
        display_name="Total Thread Density (EPI + PPI)",
        description="Derived sum of warp EPI and weft PPI from pre-production inputs.",
        data_type=DataType.FLOAT,
        unit="threads/in",
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.IMPUTE,
        constraints=FeatureConstraints(minimum=2.0),
        derived_from=("warp_ends_per_inch", "weft_picks_per_inch"),
    )

    waste_length_yds = FeatureContract(
        name="waste_length_yds",
        display_name="Rejected & Cut Waste Length (yds)",
        description="Measured rejected fabric and cut-piece waste length per beam batch.",
        data_type=DataType.FLOAT,
        unit="yd",
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("Rej_and_cut_Piece",),
    )

    target = TargetContract(
        target_id=WEAVING_TASK_ID,
        feature=waste_length_yds,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured rejected and cut-piece woven fabric length per beam batch."
            ),
            granularity="beam_batch",
            measurement_method="Post-weaving grey fabric inspection table log.",
            included_components=("cut_piece_waste", "rejected_fabric_length"),
            excluded_components=("recoverable_beam_remnant", "finishing_shrinkage"),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="weaving_waste_rate_pct",
            numerator="waste_length_yds",
            denominator="planned_beam_length_yds",
            granularity="beam_batch",
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id=WEAVING_DATASET_ID,
        version="1.0.0",
        domain="textile",
        process=WEAVING_PROCESS_ID,
        description=(
            "Pre-production weaving waste quantity prediction contract scoped at "
            "beam-batch granularity with order-level grouping."
        ),
        granularity="beam_batch",
        features=(
            batch_record_id,
            order_id,
            planned_beam_length_yds,
            planned_finish_fabric_yds,
            planned_grey_fabric_yds,
            fabric_allowance_pct,
            shrink_allowance_pct,
            warp_count_ne,
            weft_count_ne,
            warp_ends_per_inch,
            weft_picks_per_inch,
            fabric_construction,
            total_thread_density_per_inch,
        ),
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            description="Predict weaving scrap length prior to beam mounting and weaving.",
        ),
        business_keys=("batch_record_id",),
        grouping_columns=("order_id",),
    )


def build_weaving_derived_specs() -> tuple[DerivedFeatureSpec, ...]:
    """Return leakage-safe derived feature specs sliced to declared inputs."""
    return (
        DerivedFeatureSpec(
            feature_name="total_thread_density_per_inch",
            compute=lambda df: compute_weaving_thread_density_sum(
                df["warp_ends_per_inch"], df["weft_picks_per_inch"]
            ),
        ),
    )


def build_weaving_process_contract(
    *,
    confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the Weaving ProcessContract for DomainRegistry registration."""
    dataset_contract = build_weaving_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=WEAVING_TASK_ID,
        version="1.0.0",
        display_name="Pre-Production Weaving Waste Quantity Regression",
        description="Forecast expected rejected woven fabric length (yds) per beam batch.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    return ProcessContract(
        process_id=WEAVING_PROCESS_ID,
        version="1.0.0",
        display_name="Weaving",
        description="Woven fabric manufacturing process (warp prep, beam mounting, loom).",
        tasks=(task,),
    )
