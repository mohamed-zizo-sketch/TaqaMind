"""Dairy manufacturing process contract and configuration."""

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

PROCESS_ID = "dairy"
TASK_ID = "pre_production_waste_kg"

DAIRY_ALIAS_MAP: dict[str, str] = {
    "plant_id": "factory_id",
    "order_id": "batch_id",
    "scheduled_date": "production_timestamp",
    "line_id": "production_line_id",
    "dairy_product": "product_category",
    "planned_qty_kg": "planned_batch_quantity_kg",
    "pasteurization_temp_c": "planned_processing_temp_c",
    "holding_time_min": "planned_processing_duration_min",
    "milk_loss_kg": "waste_quantity_kg",
    "dairy_waste_kg": "waste_quantity_kg",
}


def build_dairy_dataset_contract(
    confirmation: TargetConfirmation | None = None,
) -> DatasetContract:
    """Build the DatasetContract for dairy pre-production waste prediction."""

    features = (
        FeatureContract(
            name="factory_id",
            display_name="Factory Identifier",
            description="Unique dairy plant identifier.",
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
            description="Unique dairy production batch identifier.",
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
            description="Dairy pasteurization/filling line identifier.",
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
            description="Dairy product family (e.g., fluid_milk, yogurt, cheese).",
            data_type=DataType.CATEGORY,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("dairy_product",),
            source_system="ERP",
        ),
        FeatureContract(
            name="planned_batch_quantity_kg",
            display_name="Planned Batch Quantity (kg)",
            description="Scheduled milk/dairy batch mass.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.RECORDED,
            unit="kg",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("planned_qty_kg",),
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            source_system="ERP",
        ),
        FeatureContract(
            name="planned_processing_temp_c",
            display_name="Planned Pasteurization Temp (degC)",
            description="Target thermal pasteurization/UHT setpoint.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="degC",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("pasteurization_temp_c",),
            constraints=FeatureConstraints(minimum=0.0, maximum=160.0),
            source_system="MES",
        ),
        FeatureContract(
            name="planned_processing_duration_min",
            display_name="Planned Processing Duration (min)",
            description="Scheduled batch thermal/fermentation cycle time.",
            data_type=DataType.FLOAT,
            role=FeatureRole.PREDICTOR,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.CONFIGURED,
            unit="min",
            required=True,
            nullable=False,
            missing_policy=MissingValuePolicy.REJECT,
            aliases=("holding_time_min",),
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            source_system="MES",
        ),
    )

    target_feature = FeatureContract(
        name="waste_quantity_kg",
        display_name="Avoidable Dairy Waste (kg)",
        description="Measured avoidable dairy product loss and rejected packs (kg).",
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit="kg",
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        aliases=("milk_loss_kg", "dairy_waste_kg"),
        constraints=FeatureConstraints(minimum=0.0),
        source_system="MES / QC Log",
    )

    target = TargetContract(
        target_id="waste_quantity_kg",
        version="1.0.0",
        feature=target_feature,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition=(
                "Measured mass (kg) of avoidable dairy loss from CIP interface push-out, "
                "overfilling, seal failures, and off-spec batch rejections."
            ),
            granularity="batch",
            measurement_method="In-line mass flowmeter / rejected pack weigh-out",
            included_components=(
                "cip_interface_milk_loss",
                "filling_spillage",
                "rejected_dairy_packs",
            ),
            excluded_components=(
                "recoverable_cheese_whey",
                "cip_rinse_water",
                "packaging_carton_scrap",
            ),
        ),
        confirmation=confirmation or TargetConfirmation(),
        derived_rate=RateDefinition(
            name="waste_rate_pct",
            numerator="waste_quantity_kg",
            denominator="planned_batch_quantity_kg",
            granularity="batch",
            output_unit="%",
            multiplier=100.0,
        ),
    )

    return DatasetContract(
        dataset_id="food_beverage_dairy_waste_v1",
        version="1.0.0",
        domain="food_beverage",
        process=PROCESS_ID,
        description="Pre-production avoidable waste contract for dairy manufacturing.",
        granularity="batch",
        features=features,
        target=target,
        prediction_context=PredictionContext(
            prediction_stage=FeatureAvailability.PRE_PRODUCTION,
            horizon_seconds=0,
            description="Pre-production prediction made before pasteurization/filling starts.",
        ),
        business_keys=("factory_id", "batch_id"),
        grouping_columns=("batch_id",),
        time_column="production_timestamp",
    )


def build_dairy_process_contract(
    confirmation: TargetConfirmation | None = None,
) -> ProcessContract:
    """Build the ProcessContract for dairy manufacturing."""

    dataset_contract = build_dairy_dataset_contract(confirmation=confirmation)
    task = IndustrialTaskContract(
        task_id=TASK_ID,
        version="1.0.0",
        display_name="Dairy Pre-Production Waste Prediction (kg)",
        description="Predict avoidable batch waste mass (kg) prior to dairy production.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset_contract,
    )
    return ProcessContract(
        process_id=PROCESS_ID,
        version="1.0.0",
        display_name="Dairy Manufacturing",
        description="Fluid milk, yogurt, and cheese industrial processing.",
        tasks=(task,),
    )
