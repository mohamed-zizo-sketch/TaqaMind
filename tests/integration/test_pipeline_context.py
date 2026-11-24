"""Integration tests for execution context and industrial contracts."""

from uuid import uuid4

import pytest

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
)
from taqamind.core.interfaces import PipelineContext


def test_context_created_from_industrial_task(
    textile_domain,
):
    task = textile_domain.get_task(
        "knitting",
        "waste_prediction",
    )

    context = PipelineContext.from_task(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        task=task,
        purpose=DatasetPurpose.TRAINING,
    )

    assert context.domain == "textile"
    assert context.process == "knitting"
    assert context.task_id == task.task_id

    assert (
        context.contract_version
        == task.dataset_contract.version
    )

    context.validate_task_contract(task)


def test_context_rejects_different_domain(
    textile_domain,
    food_domain,
):
    textile_task = textile_domain.get_task(
        "knitting",
        "waste_prediction",
    )

    food_task = food_domain.get_task(
        "synthetic_food_process",
        "waste_prediction",
    )

    context = PipelineContext.from_task(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        task=textile_task,
        purpose=DatasetPurpose.TRAINING,
    )

    with pytest.raises(
        ValueError,
        match="mismatch for 'domain'",
    ):
        context.validate_task_contract(food_task)


def test_context_rejects_different_process(
    textile_domain,
    weaving_domain,
):
    knitting = textile_domain.get_task(
        "knitting",
        "waste_prediction",
    )

    weaving = weaving_domain.get_task(
        "weaving",
        "waste_prediction",
    )

    context = PipelineContext.from_task(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        task=knitting,
        purpose=DatasetPurpose.TRAINING,
    )

    with pytest.raises(
        ValueError,
        match="mismatch for 'process'",
    ):
        context.validate_task_contract(weaving)


def test_context_rejects_different_contract_version(
    textile_domain,
):
    task = textile_domain.get_task(
        "knitting",
        "waste_prediction",
    )

    context = PipelineContext.from_task(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        task=task,
        purpose=DatasetPurpose.TRAINING,
    )

    payload = task.dataset_contract.model_dump()
    payload["version"] = "2.0.0"

    updated_contract = DatasetContract.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="mismatch for 'version'",
    ):
        context.validate_dataset_contract(
            updated_contract
        )


def test_context_rejects_different_target_version(
    textile_domain,
):
    task = textile_domain.get_task(
        "knitting",
        "waste_prediction",
    )

    context = PipelineContext.from_task(
        factory_id=uuid4(),
        dataset_revision_id=uuid4(),
        task=task,
        purpose=DatasetPurpose.TRAINING,
    )

    payload = task.dataset_contract.model_dump()
    payload["target"]["version"] = "2.0.0"

    updated_contract = DatasetContract.model_validate(
        payload
    )

    with pytest.raises(
        ValueError,
        match="target version mismatch",
    ):
        context.validate_dataset_contract(
            updated_contract
        )


def test_all_domains_use_same_execution_context(
    all_domains,
):
    for domain in all_domains:
        task = domain.processes[0].tasks[0]

        context = PipelineContext.from_task(
            factory_id=uuid4(),
            dataset_revision_id=uuid4(),
            task=task,
            purpose=DatasetPurpose.TRAINING,
        )

        assert context.domain == domain.domain_id

        context.validate_task_contract(task)