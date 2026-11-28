"""Beverage manufacturing process contract and configuration."""

from __future__ import annotations

from taqamind.core.contracts import (
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

PROCESS_ID = "beverage"
TASK_ID = "pre_production_waste_l"

BEVERAGE_ALIAS_MAP: dict[str, str] = {
    "plant_id": "factory_id",
    "order_id": "batch_id",
    "scheduled_date": "production_timestamp",
    "line_id": "production_line_id",
    "beverage_style": "product_category",
    "planned_volume_l": "planned_batch_volume_l",
    "batch_volume_l": "planned_batch_volume_l",
    "setpoint_temp_c": "planned_processing_temp_c",
    "cycle_time_min": "planned_processing_duration_min",
    "liquid_loss_l": "waste_quantity_l",
    "spillage_l": "waste_quantity_l",
}


def build_beverage_dataset_contract(
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the DatasetContract for beverage pre-production waste prediction."""

    features = (
        FeatureContract(
            name="factory_id",
            display_name="Factory Identifier",
            description="Unique beverage plant identifier.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.STATIC,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("plant_id",),
            source_system="ERP",
        ),
        FeatureContract(
            name="batch_id",
            display_name="Batch Identifier",
            description="Unique beverage batch identifier.",
            data_type=DataType.STRING,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("order_id",),
            source_system="ERP",
        ),
        FeatureContract(
            name="production_timestamp",
            display_name="Planned Production Timestamp",
            description="Scheduled batch start timestamp.",
            data_type=DataType.DATETIME,
            role=FeatureRole.TIMESTAMP,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("scheduled_date",),
            source_system="ERP",
        ),
        FeatureContract(
            name="production_line_id",
            display_name="Production Line ID",
            description="Beverage blending/filling line identifier.",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("line_id",),
            source_system="MES",
        ),
        FeatureContract(
            name="product_category",
            display_name="Product Category",
            description="Beverage family (carbonated_soft_drink, juice, bottled_water, beer).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("beverage_style",),
            source_system="ERP",
        ),
        FeatureContract(
            name="planned_batch_volume_l",
            display_name="Planned Batch Volume (L)",
            description="Scheduled liquid volume for the batch.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="L",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("planned_volume_l", "batch_volume_l"),
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            source_system="ERP",
        ),
        FeatureContract(
            name="planned_processing_temp_c",
            display_name="Planned Processing Temperature (degC)",
            description="Target blending/fermentation/pasteurization setpoint temperature.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="degC",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("setpoint_temp_c",),
            constraints=FeatureConstraints(minimum=-5.0, maximum=140.0),
            source_system="MES",
        ),
        FeatureContract(
            name="planned_processing_duration_min",
            display_name="Planned Processing Duration (min)",
            description="Scheduled batch cycle duration.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="min",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("cycle_time_min",),
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            source_system="MES",
        ),
        FeatureContract(
            name="target_ph_level",
            display_name="Target Recipe pH",
            description="Target pH specification before production.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="pH",
            required=False,
            nullable=True,
            missing_policy=MissingValuePolicy.IMPUTE,
            constraints=FeatureConstraints(minimum=2.0, maximum=9.0),
            source_system="Recipe Master",
        ),
    )

    target_feature = FeatureContract(
        name="waste_quantity_l",
        display_name="Avoidable Liquid Waste (L)",
        description="Measured avoidable beverage spillage, purge, and rejected liquid (L).",
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit="L",
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("liquid_loss_l", "spillage_l"),
        constraints=FeatureConstraints(minimum=0.0),
        source_system="Flowmeter / Filler Reject Log",
    )

    target = TargetContract(
        target_id="waste_quantity_l",
        version="1.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured liquid volume (L) of avoidable beverage loss from line purge, "
                "filler overfill/foaming, bottle/can breakage liquid, and rejected units."
            ),
            granularity="batch",
            measurement_method="Differential tank/filler flowmeter and reject drain counter",
            included_components=(
                "line_start_stop_purge",
                "filler_spillage_and_overfill",
                "rejected_filled_containers",
            ),
            excluded_components=(
                "recoverable_spent_grain",
                "recoverable_yeast_byproduct",
                "empty_bottle_or_can_scrap",
            ),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="waste_rate_pct",
            numerator="waste_quantity_l",
            denominator="planned_batch_volume_l",
            granularity="batch",
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id="food_beverage_beverage_waste_v1",
        version="1.0.0",
        domain="food_beverage",
        process=PROCESS_ID,
        description="Pre-production avoidable waste contract for beverage manufacturing.",
        granularity="batch",
        features=features,
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            horizon_seconds=0,
            description="Pre-production prediction made before syrup blending/filling starts.",
        ),
        business_keys=("factory_id", "batch_id"),
        grouping_columns=("batch_id",),
        time_column="production_timestamp",
    )


def build_beverage_process_contract(
    confirmation: TargetConfirmation | None = None,
    energy_confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the ProcessContract for beverage manufacturing."""
    from taqamind.domains.food_beverage.energy_contracts import (
        build_beverage_energy_task_contract,
    )

    dataset_contract = build_beverage_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=TASK_ID,
        version="1.0.0",
        display_name="Beverage Pre-Production Waste Prediction (L)",
        description="Predict avoidable batch liquid waste (L) prior to beverage production.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    energy_task = build_beverage_energy_task_contract(confirmation=energy_confirmation)
    return ProcessContract(
        process_id=PROCESS_ID,
        version="1.0.0",
        display_name="Beverage Manufacturing",
        description="Soft drinks, juices, bottled water, and brewing manufacturing.",
        tasks=(task, energy_task),
    )
