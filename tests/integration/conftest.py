"""Synthetic industrial contracts for integration testing."""

import pytest

from taqamind.core.contracts import (
    DatasetContract,
    DataType,
    DomainContract,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    IndustrialTaskContract,
    PlatformModule,
    PredictionContext,
    ProcessContract,
    RateDefinition,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)


def build_domain(
    domain_id: str,
    process_id: str,
    predictor_name: str,
    target_name: str,
    unit: str,
) -> DomainContract:
    """Build a synthetic industrial domain."""

    production_quantity = FeatureContract(
        name=predictor_name,
        display_name="Planned Quantity",
        description="Quantity known before production.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
        unit=unit,
        constraints=FeatureConstraints(
            minimum=0,
            minimum_inclusive=False,
        ),
    )

    run_id = FeatureContract(
        name="run_id",
        display_name="Production Run",
        description="Synthetic production run identifier.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
    )

    order_id = FeatureContract(
        name="order_id",
        display_name="Production Order",
        description="Identifier for related production runs.",
        data_type=DataType.STRING,
        role=FeatureRole.IDENTIFIER,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
    )

    planned_start = FeatureContract(
        name="planned_start",
        display_name="Planned Start",
        description="Scheduled production start time.",
        data_type=DataType.DATETIME,
        role=FeatureRole.TIMESTAMP,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.RECORDED,
    )

    waste = FeatureContract(
        name=target_name,
        display_name="Waste Quantity",
        description="Measured waste after production.",
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit=unit,
        constraints=FeatureConstraints(minimum=0),
    )

    target = TargetContract(
        target_id="waste_prediction",
        feature=waste,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(
            definition="Synthetic measured production waste.",
            granularity="production_run",
        ),
        derived_rate=RateDefinition(
            name="waste_rate_pct",
            numerator=target_name,
            denominator=predictor_name,
            granularity="production_run",
        ),
    )

    dataset = DatasetContract(
        dataset_id=f"{domain_id}_{process_id}_waste",
        domain=domain_id,
        process=process_id,
        description="Synthetic integration dataset.",
        granularity="production_run",
        features=(
            production_quantity,
            run_id,
            order_id,
            planned_start,
        ),
        target=target,
        prediction_context=PredictionContext(
            description="Predict waste before production.",
        ),
        business_keys=("run_id",),
        grouping_columns=("order_id",),
        time_column="planned_start",
    )

    task = IndustrialTaskContract(
        task_id="waste_prediction",
        display_name="Waste Prediction",
        description="Synthetic pre-production waste task.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=TaskType.REGRESSION,
        dataset_contract=dataset,
    )

    process = ProcessContract(
        process_id=process_id,
        display_name=process_id.replace("_", " ").title(),
        description="Synthetic manufacturing process.",
        tasks=(task,),
    )

    return DomainContract(
        domain_id=domain_id,
        display_name=domain_id.replace("_", " ").title(),
        description="Synthetic industrial domain.",
        processes=(process,),
    )


@pytest.fixture
def textile_domain() -> DomainContract:
    return build_domain(
        domain_id="textile",
        process_id="knitting",
        predictor_name="planned_fabric_weight_kg",
        target_name="actual_fabric_waste_kg",
        unit="kg",
    )


@pytest.fixture
def food_domain() -> DomainContract:
    return build_domain(
        domain_id="food_beverage",
        process_id="synthetic_food_process",
        predictor_name="planned_batch_weight_kg",
        target_name="actual_food_waste_kg",
        unit="kg",
    )


@pytest.fixture
def iron_domain() -> DomainContract:
    return build_domain(
        domain_id="iron_manufacturing",
        process_id="synthetic_iron_process",
        predictor_name="planned_metal_weight_kg",
        target_name="actual_metal_scrap_kg",
        unit="kg",
    )


@pytest.fixture
def weaving_domain() -> DomainContract:
    return build_domain(
        domain_id="textile",
        process_id="weaving",
        predictor_name="planned_fabric_length_m",
        target_name="actual_fabric_waste_m",
        unit="m",
    )


@pytest.fixture
def all_domains(
    textile_domain,
    food_domain,
    iron_domain,
) -> tuple[DomainContract, ...]:
    return (
        textile_domain,
        food_domain,
        iron_domain,
    )


@pytest.fixture
def textile_dataset(textile_domain) -> DatasetContract:
    return textile_domain.get_dataset_contract(
        "knitting",
        "waste_prediction",
    )