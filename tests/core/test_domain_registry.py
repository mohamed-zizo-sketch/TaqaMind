"""Tests for industrial domain registration and discovery."""

import pytest

from taqamind.core.contracts.domain import (
    DomainContract,
    IndustrialTaskContract,
    ProcessContract,
)
from taqamind.core.contracts.enums import PlatformModule
from taqamind.core.registry import (
    DomainAlreadyRegisteredError,
    DomainNotFoundError,
    DomainRegistry,
)
from tests.core.interfaces.test_readiness import make_contract


def build_test_domain(
    domain_id="textile",
    process_id="knitting",
    version="1.0.0",
):
    """Create a synthetic domain for registry tests only."""

    dataset_payload = make_contract().model_dump()

    dataset_payload.update(
        {
            "domain": domain_id,
            "process": process_id,
            "dataset_id": f"{domain_id}_{process_id}_waste",
        }
    )

    from taqamind.core.contracts.dataset import DatasetContract

    dataset = DatasetContract.model_validate(
        dataset_payload
    )

    task = IndustrialTaskContract(
        task_id="waste_prediction",
        display_name="Waste Prediction",
        description="Synthetic waste prediction task.",
        module=PlatformModule.WASTE_MANAGEMENT,
        task_type=dataset.target.task_type,
        dataset_contract=dataset,
    )

    process = ProcessContract(
        process_id=process_id,
        display_name=process_id.replace("_", " ").title(),
        description="Synthetic industrial process.",
        tasks=(task,),
    )

    return DomainContract(
        domain_id=domain_id,
        version=version,
        display_name=domain_id.replace("_", " ").title(),
        description="Synthetic domain contract.",
        processes=(process,),
    )


def test_empty_registry():
    registry = DomainRegistry()

    assert registry.registration_count == 0
    assert registry.domain_ids == ()
    assert registry.list_domains() == ()


def test_register_domain():
    registry = DomainRegistry()
    domain = build_test_domain()

    registered = registry.register(domain)

    assert registered == domain
    assert registry.registration_count == 1


def test_get_registered_domain():
    registry = DomainRegistry()
    domain = build_test_domain()

    registry.register(domain)

    assert registry.get_domain(
        "textile",
        "1.0.0",
    ) == domain


def test_unknown_domain_rejected():
    registry = DomainRegistry()

    with pytest.raises(DomainNotFoundError):
        registry.get_domain(
            "unknown_domain",
            "1.0.0",
        )


def test_identical_registration_is_idempotent():
    registry = DomainRegistry()
    domain = build_test_domain()

    registry.register(domain)
    registry.register(domain)

    assert registry.registration_count == 1


def test_conflicting_registration_rejected():
    registry = DomainRegistry()
    original = build_test_domain()

    registry.register(original)

    payload = original.model_dump()
    payload["description"] = "Changed domain definition."

    conflicting = DomainContract.model_validate(
        payload
    )

    with pytest.raises(DomainAlreadyRegisteredError):
        registry.register(conflicting)


def test_multiple_domain_versions():
    registry = DomainRegistry()

    first = build_test_domain(version="1.0.0")
    second = build_test_domain(version="2.0.0")

    registry.register(first)
    registry.register(second)

    assert registry.registration_count == 2

    assert registry.domain_ids == ("textile",)

    assert registry.list_versions("textile") == (
        "1.0.0",
        "2.0.0",
    )


def test_numeric_version_order():
    registry = DomainRegistry()

    registry.register(
        build_test_domain(version="1.10.0")
    )

    registry.register(
        build_test_domain(version="1.2.0")
    )

    assert registry.list_versions("textile") == (
        "1.2.0",
        "1.10.0",
    )


def test_multiple_industrial_domains():
    registry = DomainRegistry()

    specifications = (
        ("textile", "knitting"),
        ("food_beverage", "synthetic_food_process"),
        ("iron_manufacturing", "synthetic_iron_process"),
    )

    for domain_id, process_id in specifications:
        registry.register(
            build_test_domain(
                domain_id=domain_id,
                process_id=process_id,
            )
        )

    assert registry.registration_count == 3

    assert registry.domain_ids == (
        "food_beverage",
        "iron_manufacturing",
        "textile",
    )


def test_get_process():
    registry = DomainRegistry()

    registry.register(build_test_domain())

    process = registry.get_process(
        domain_id="textile",
        domain_version="1.0.0",
        process_id="knitting",
    )

    assert process.process_id == "knitting"


def test_get_task():
    registry = DomainRegistry()

    registry.register(build_test_domain())

    task = registry.get_task(
        domain_id="textile",
        domain_version="1.0.0",
        process_id="knitting",
        task_id="waste_prediction",
    )

    assert task.task_id == "waste_prediction"


def test_get_dataset_contract():
    registry = DomainRegistry()

    registry.register(build_test_domain())

    contract = registry.get_dataset_contract(
        domain_id="textile",
        domain_version="1.0.0",
        process_id="knitting",
        task_id="waste_prediction",
    )

    assert contract.domain == "textile"
    assert contract.process == "knitting"


def test_find_tasks_by_module():
    registry = DomainRegistry()

    registry.register(
        build_test_domain(
            "textile",
            "knitting",
        )
    )

    registry.register(
        build_test_domain(
            "food_beverage",
            "synthetic_food_process",
        )
    )

    results = registry.find_tasks_by_module(
        PlatformModule.WASTE_MANAGEMENT
    )

    assert len(results) == 2

    assert {
        item[0] for item in results
    } == {
        "textile",
        "food_beverage",
    }


def test_unregistered_module_returns_empty():
    registry = DomainRegistry()

    registry.register(build_test_domain())

    assert registry.find_tasks_by_module(
        PlatformModule.ENERGY_EFFICIENCY
    ) == ()


def test_new_domain_requires_no_registry_changes():
    registry = DomainRegistry()

    new_domain = build_test_domain(
        domain_id="synthetic_new_industry",
        process_id="synthetic_process",
    )

    registry.register(new_domain)

    assert registry.has_domain(
        "synthetic_new_industry",
        "1.0.0",
    )