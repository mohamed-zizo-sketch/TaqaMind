"""End-to-end integration tests for shared industrial contracts."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    DomainContract,
    FeatureAvailability,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    IssueImpact,
    PipelineStage,
    PlatformModule,
    ReadinessStatus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


def rebuild_dataset(dataset, **changes):
    """Revalidate a dataset after changing its definition."""

    payload = dataset.model_dump()
    payload.update(changes)

    return DatasetContract.model_validate(payload)


def make_report(dataset, revision_id, stage, issues=()):
    """Build a validation report for a dataset contract."""

    return ValidationReport(
        dataset_id=dataset.dataset_id,
        contract_version=dataset.version,
        dataset_revision_id=revision_id,
        domain=dataset.domain,
        process=dataset.process,
        purpose=DatasetPurpose.TRAINING,
        completed_stages=(stage,),
        issues=issues,
    )


def test_all_domains_use_shared_contracts(all_domains):
    assert len(all_domains) == 3

    for domain in all_domains:
        assert isinstance(domain, DomainContract)

        for process in domain.processes:
            for task in process.tasks:
                assert isinstance(
                    task.dataset_contract,
                    DatasetContract,
                )


def test_domain_identifiers_are_unique(all_domains):
    identifiers = [
        domain.domain_id
        for domain in all_domains
    ]

    assert len(identifiers) == len(set(identifiers))


def test_same_task_across_domains(all_domains):
    for domain in all_domains:
        process = domain.processes[0]

        task = domain.get_task(
            process.process_id,
            "waste_prediction",
        )

        assert task.module == PlatformModule.WASTE_MANAGEMENT
        assert task.dataset_contract.domain == domain.domain_id


def test_domain_specific_predictors_are_independent(all_domains):
    predictors = set()

    for domain in all_domains:
        dataset = domain.processes[0].tasks[0].dataset_contract

        predictors.update(dataset.predictor_names)

    assert len(predictors) == 3


def test_training_schema_across_domains(all_domains):
    for domain in all_domains:
        dataset = domain.processes[0].tasks[0].dataset_contract

        required = dataset.required_columns(
            DatasetPurpose.TRAINING
        )

        assert dataset.target.feature.name in required
        assert "run_id" in required
        assert "order_id" in required
        assert "planned_start" in required


def test_prediction_schema_excludes_targets(all_domains):
    for domain in all_domains:
        dataset = domain.processes[0].tasks[0].dataset_contract

        required = dataset.required_columns(
            DatasetPurpose.PREDICTION
        )

        assert dataset.target.feature.name not in required

        for name in dataset.predictor_names:
            assert name in required


def test_identifiers_are_not_model_predictors(textile_dataset):
    assert "run_id" not in textile_dataset.predictor_names
    assert "order_id" not in textile_dataset.predictor_names
    assert "planned_start" not in textile_dataset.predictor_names


def test_business_key_and_grouping_metadata(textile_dataset):
    assert textile_dataset.business_keys == ("run_id",)
    assert textile_dataset.grouping_columns == ("order_id",)
    assert textile_dataset.time_column == "planned_start"


def test_textile_processes_support_different_units(
    textile_domain,
    weaving_domain,
):
    knitting = textile_domain.get_dataset_contract(
        "knitting",
        "waste_prediction",
    )

    weaving = weaving_domain.get_dataset_contract(
        "weaving",
        "waste_prediction",
    )

    assert knitting.target.feature.unit == "kg"
    assert weaving.target.feature.unit == "m"


def test_combined_textile_domain(
    textile_domain,
    weaving_domain,
):
    domain = DomainContract(
        domain_id="textile",
        display_name="Textile Manufacturing",
        description="Synthetic textile processes.",
        processes=(
            *textile_domain.processes,
            *weaving_domain.processes,
        ),
    )

    assert domain.total_tasks == 2

    assert set(domain.process_ids) == {
        "knitting",
        "weaving",
    }


@pytest.mark.parametrize(
    "availability",
    [
        FeatureAvailability.IN_PRODUCTION,
        FeatureAvailability.POST_PRODUCTION,
        FeatureAvailability.UNKNOWN,
    ],
)
def test_unavailable_predictor_rejected(
    textile_dataset,
    availability,
):
    unsafe_feature = FeatureContract(
        name="unsafe_measurement",
        display_name="Unsafe Measurement",
        description="Unavailable at prediction time.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=availability,
    )

    with pytest.raises(ValidationError):
        rebuild_dataset(
            textile_dataset,
            features=(
                *textile_dataset.features,
                unsafe_feature,
            ),
        )


def test_target_derived_predictor_rejected(textile_dataset):
    leaked_feature = FeatureContract(
        name="leaked_waste",
        display_name="Leaked Waste",
        description="Feature derived from actual waste.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        derived_from=(
            textile_dataset.target.feature.name,
        ),
    )

    with pytest.raises(ValidationError):
        rebuild_dataset(
            textile_dataset,
            features=(
                *textile_dataset.features,
                leaked_feature,
            ),
        )


def test_valid_derived_predictor(textile_dataset):
    source = textile_dataset.predictor_names[0]

    derived = FeatureContract(
        name="planned_weight_tonnes",
        display_name="Planned Weight",
        description="Planned weight converted to tonnes.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        unit="t",
        derived_from=(source,),
    )

    dataset = rebuild_dataset(
        textile_dataset,
        features=(
            *textile_dataset.features,
            derived,
        ),
    )

    assert derived.name in dataset.predictor_names

    assert dataset.get_dependencies(
        derived.name
    ) == (source,)

    required = dataset.required_columns(
        DatasetPurpose.PREDICTION
    )

    assert source in required
    assert derived.name not in required


def test_domain_contract_serialization(all_domains):
    for domain in all_domains:
        restored = DomainContract.model_validate_json(
            domain.model_dump_json()
        )

        assert restored == domain


def test_dataset_versioning(textile_dataset):
    updated = rebuild_dataset(
        textile_dataset,
        version="2.0.0",
    )

    assert textile_dataset.version == "1.0.0"
    assert updated.version == "2.0.0"

    assert (
        updated.dataset_id
        == textile_dataset.dataset_id
    )


def test_contracts_cannot_be_modified_directly(textile_dataset):
    with pytest.raises(ValidationError):
        textile_dataset.version = "2.0.0"


def test_unconfirmed_target_remains_unconfirmed(all_domains):
    for domain in all_domains:
        dataset = domain.processes[0].tasks[0].dataset_contract

        assert not dataset.target.is_confirmed


def test_validation_reports_merge_correctly(textile_dataset):
    revision = uuid4()

    ingestion = make_report(
        textile_dataset,
        revision,
        PipelineStage.INGESTION,
    )

    profiling = make_report(
        textile_dataset,
        revision,
        PipelineStage.PROFILING,
        issues=(
            ValidationIssue(
                code="DUPLICATE_RECORD_CANDIDATE",
                stage=PipelineStage.PROFILING,
                severity=ValidationSeverity.WARNING,
                impact=IssueImpact.LIMITED,
                message="Potential duplicate records.",
            ),
        ),
    )

    combined = ValidationReport.merge(
        (ingestion, profiling)
    )

    assert combined.status == ReadinessStatus.LIMITED
    assert combined.summary.total_issues == 1
    assert len(combined.completed_stages) == 2


def test_blocking_report_overrides_ready(textile_dataset):
    revision = uuid4()

    ingestion = make_report(
        textile_dataset,
        revision,
        PipelineStage.INGESTION,
    )

    semantic = make_report(
        textile_dataset,
        revision,
        PipelineStage.SEMANTIC_VALIDATION,
        issues=(
            ValidationIssue(
                code="INVALID_TARGET",
                stage=PipelineStage.SEMANTIC_VALIDATION,
                severity=ValidationSeverity.ERROR,
                impact=IssueImpact.BLOCKING,
                message="Invalid target values detected.",
            ),
        ),
    )

    combined = ValidationReport.merge(
        (ingestion, semantic)
    )

    assert combined.status == ReadinessStatus.BLOCKED
    assert combined.has_blocking_issues


def test_reports_from_different_revisions_cannot_merge(
    textile_dataset,
):
    first = make_report(
        textile_dataset,
        uuid4(),
        PipelineStage.INGESTION,
    )

    second = make_report(
        textile_dataset,
        uuid4(),
        PipelineStage.PROFILING,
    )

    with pytest.raises(ValueError, match="incompatible"):
        ValidationReport.merge(
            (first, second)
        )


def test_reports_from_different_domains_cannot_merge(
    textile_dataset,
    food_domain,
):
    revision = uuid4()

    food_dataset = food_domain.get_dataset_contract(
        "synthetic_food_process",
        "waste_prediction",
    )

    textile_report = make_report(
        textile_dataset,
        revision,
        PipelineStage.INGESTION,
    )

    food_report = make_report(
        food_dataset,
        revision,
        PipelineStage.PROFILING,
    )

    with pytest.raises(ValueError, match="incompatible"):
        ValidationReport.merge(
            (textile_report, food_report)
        )


def test_report_serialization(textile_dataset):
    report = make_report(
        textile_dataset,
        uuid4(),
        PipelineStage.INGESTION,
    )

    restored = ValidationReport.model_validate(
        report.to_payload()
    )

    assert restored == report


def test_rate_definition_uses_target_and_correct_unit(
    textile_dataset,
):
    rate = textile_dataset.target.derived_rate

    assert rate is not None

    denominator = textile_dataset.get_feature(
        rate.denominator
    )

    assert rate.numerator == textile_dataset.target.feature.name

    assert (
        denominator.unit
        == textile_dataset.target.feature.unit
    )


def test_ready_stage_is_not_training_authorization(
    textile_dataset,
):
    report = make_report(
        textile_dataset,
        uuid4(),
        PipelineStage.INGESTION,
    )

    assert report.status == ReadinessStatus.READY
    assert not textile_dataset.target.is_confirmed

    assert PipelineStage.READINESS not in report.completed_stages