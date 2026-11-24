"""Tests for hierarchical industrial domain contracts."""

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetContract,
    DataType,
    DomainContract,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureRole,
    IndustrialTaskContract,
    PlatformModule,
    PredictionContext,
    ProcessContract,
    TargetContract,
    TargetDefinition,
    TargetKind,
    TaskType,
)


def make_dataset(
    domain="textile",
    process="knitting",
    **overrides,
):
    data = {
        "dataset_id": f"{domain}_{process}_waste",
        "domain": domain,
        "process": process,
        "description": "Synthetic industrial dataset.",
        "granularity": "production_run",
        "features": (
            FeatureContract(
                name="planned_quantity_kg",
                display_name="Planned Quantity",
                description="Planned production quantity.",
                data_type=DataType.FLOAT,
                role=FeatureRole.PREDICTOR,
                availability=FeatureAvailability.PRE_PRODUCTION,
                unit="kg",
            ),
        ),
        "target": TargetContract(
            target_id="waste_prediction",
            feature=FeatureContract(
                name="actual_waste_kg",
                display_name="Actual Waste",
                description="Measured production waste.",
                data_type=DataType.FLOAT,
                role=FeatureRole.TARGET,
                availability=FeatureAvailability.POST_PRODUCTION,
                unit="kg",
                constraints=FeatureConstraints(minimum=0),
            ),
            task_type=TaskType.REGRESSION,
            target_kind=TargetKind.QUANTITY,
            definition=TargetDefinition(
                definition="Measured production waste.",
                granularity="production_run",
            ),
        ),
        "prediction_context": PredictionContext(
            description="Predict before production.",
        ),
    }

    data.update(overrides)

    return DatasetContract(**data)


def make_task(
    domain="textile",
    process="knitting",
    **overrides,
):
    data = {
        "task_id": "waste_prediction",
        "display_name": "Waste Prediction",
        "description": "Predict industrial waste.",
        "module": PlatformModule.WASTE_MANAGEMENT,
        "task_type": TaskType.REGRESSION,
        "dataset_contract": make_dataset(
            domain=domain,
            process=process,
        ),
    }

    data.update(overrides)

    return IndustrialTaskContract(**data)


def make_process(
    domain="textile",
    process="knitting",
    **overrides,
):
    data = {
        "process_id": process,
        "display_name": process.title(),
        "description": f"Industrial {process} process.",
        "tasks": (
            make_task(
                domain=domain,
                process=process,
            ),
        ),
    }

    data.update(overrides)

    return ProcessContract(**data)


def make_domain(**overrides):
    data = {
        "domain_id": "textile",
        "display_name": "Textile Manufacturing",
        "description": "Industrial textile manufacturing.",
        "processes": (
            make_process(process="knitting"),
            make_process(process="weaving"),
        ),
    }

    data.update(overrides)

    return DomainContract(**data)


def test_valid_industrial_task():
    task = make_task()

    assert task.task_id == "waste_prediction"
    assert task.task_type == TaskType.REGRESSION


def test_task_type_mismatch():
    with pytest.raises(
        ValidationError,
        match="task type must match",
    ):
        make_task(
            task_type=TaskType.CLASSIFICATION,
        )


def test_valid_process():
    process = make_process()

    assert process.process_id == "knitting"

    assert process.task_ids == (
        "waste_prediction",
    )


def test_empty_process_tasks():
    with pytest.raises(ValidationError):
        make_process(tasks=())


def test_duplicate_tasks():
    task = make_task()

    with pytest.raises(
        ValidationError,
        match="Task identifiers must be unique",
    ):
        make_process(tasks=(task, task))


def test_incompatible_dataset_process():
    with pytest.raises(
        ValidationError,
        match="incompatible dataset process",
    ):
        make_process(
            tasks=(
                make_task(process="weaving"),
            ),
        )


def test_process_task_lookup():
    process = make_process()

    task = process.get_task("waste_prediction")

    assert task.module == PlatformModule.WASTE_MANAGEMENT


def test_unknown_process_task():
    process = make_process()

    with pytest.raises(KeyError):
        process.get_task("unknown_task")


def test_valid_domain():
    domain = make_domain()

    assert domain.domain_id == "textile"

    assert domain.process_ids == (
        "knitting",
        "weaving",
    )

    assert domain.total_tasks == 2


def test_empty_domain_processes():
    with pytest.raises(ValidationError):
        make_domain(processes=())


def test_duplicate_processes():
    process = make_process()

    with pytest.raises(
        ValidationError,
        match="Process identifiers must be unique",
    ):
        make_domain(processes=(process, process))


def test_incompatible_dataset_domain():
    with pytest.raises(
        ValidationError,
        match="incompatible dataset domain",
    ):
        make_domain(
            processes=(
                make_process(domain="food_beverage"),
            ),
        )


def test_domain_process_lookup():
    domain = make_domain()

    process = domain.get_process("weaving")

    assert process.process_id == "weaving"


def test_unknown_domain_process():
    domain = make_domain()

    with pytest.raises(KeyError):
        domain.get_process("unknown_process")


def test_domain_task_lookup():
    domain = make_domain()

    task = domain.get_task(
        process_id="knitting",
        task_id="waste_prediction",
    )

    assert task.dataset_contract.process == "knitting"


def test_domain_dataset_lookup():
    domain = make_domain()

    dataset = domain.get_dataset_contract(
        process_id="knitting",
        task_id="waste_prediction",
    )

    assert dataset.domain == "textile"
    assert dataset.process == "knitting"

    assert dataset.target.feature.name == (
        "actual_waste_kg"
    )


def test_supported_modules():
    domain = make_domain()

    assert domain.supported_modules == (
        PlatformModule.WASTE_MANAGEMENT,
    )


def test_filter_process_tasks_by_module():
    process = make_process()

    tasks = process.get_tasks_by_module(
        PlatformModule.WASTE_MANAGEMENT
    )

    assert len(tasks) == 1


def test_filter_domain_tasks_by_module():
    domain = make_domain()

    tasks = domain.get_tasks_by_module(
        PlatformModule.WASTE_MANAGEMENT
    )

    assert len(tasks) == 2

    assert {process_id for process_id, _ in tasks} == {
        "knitting",
        "weaving",
    }


def test_unregistered_module_returns_empty_result():
    domain = make_domain()

    tasks = domain.get_tasks_by_module(
        PlatformModule.ENERGY_EFFICIENCY
    )

    assert tasks == ()


def test_same_task_id_in_different_processes():
    domain = make_domain()

    knitting_task = domain.get_task(
        "knitting",
        "waste_prediction",
    )

    weaving_task = domain.get_task(
        "weaving",
        "waste_prediction",
    )

    assert knitting_task.task_id == weaving_task.task_id

    assert (
        knitting_task.dataset_contract.process
        != weaving_task.dataset_contract.process
    )


def test_conflicting_dataset_definitions():
    first_dataset = make_dataset()

    second_dataset = make_dataset(
        description="A conflicting dataset definition.",
    )

    with pytest.raises(
        ValidationError,
        match="Conflicting dataset definitions",
    ):
        make_domain(
            processes=(
                make_process(
                    tasks=(
                        make_task(
                            task_id="task_one",
                            dataset_contract=first_dataset,
                        ),
                        make_task(
                            task_id="task_two",
                            dataset_contract=second_dataset,
                        ),
                    ),
                ),
            ),
        )


def test_identical_dataset_contract_can_be_reused():
    dataset = make_dataset()

    domain = make_domain(
        processes=(
            make_process(
                tasks=(
                    make_task(
                        task_id="task_one",
                        dataset_contract=dataset,
                    ),
                    make_task(
                        task_id="task_two",
                        dataset_contract=dataset,
                    ),
                ),
            ),
        ),
    )

    assert domain.total_tasks == 2


def test_distinct_dataset_versions_are_allowed():
    domain = make_domain(
        processes=(
            make_process(
                tasks=(
                    make_task(
                        task_id="task_one",
                        dataset_contract=make_dataset(
                            version="1.0.0",
                        ),
                    ),
                    make_task(
                        task_id="task_two",
                        dataset_contract=make_dataset(
                            version="2.0.0",
                        ),
                    ),
                ),
            ),
        ),
    )

    assert domain.total_tasks == 2


@pytest.mark.parametrize(
    ("domain_id", "process_id"),
    [
        ("textile", "knitting"),
        ("food_beverage", "synthetic_food_process"),
        ("iron_manufacturing", "synthetic_iron_process"),
    ],
)
def test_multi_domain_compatibility(
    domain_id,
    process_id,
):
    domain = DomainContract(
        domain_id=domain_id,
        display_name=domain_id,
        description="Synthetic domain for testing.",
        processes=(
            make_process(
                domain=domain_id,
                process=process_id,
            ),
        ),
    )

    dataset = domain.get_dataset_contract(
        process_id=process_id,
        task_id="waste_prediction",
    )

    assert dataset.domain == domain_id
    assert dataset.process == process_id


def test_invalid_domain_identifier():
    with pytest.raises(ValidationError):
        make_domain(domain_id="Invalid Domain")


def test_invalid_domain_version():
    with pytest.raises(ValidationError):
        make_domain(version="invalid")


def test_domain_is_frozen():
    domain = make_domain()

    with pytest.raises(ValidationError):
        domain.domain_id = "another_domain"


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        make_domain(unknown_field=True)


def test_domain_json_round_trip():
    domain = make_domain()

    restored = DomainContract.model_validate_json(
        domain.model_dump_json()
    )

    assert restored == domain