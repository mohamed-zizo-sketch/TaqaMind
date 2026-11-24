"""Tests for industrial processing output models."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from taqamind.core.interfaces import (
    ApprovedMapping,
    ColumnMappingEntry,
    ColumnProfile,
    DatasetProfile,
    MappingMethod,
    MappingProposal,
)


def make_column(**overrides):
    data = {
        "name": "Planned Qty",
        "physical_dtype": "float64",
        "missing_count": 10,
        "unique_count": 50,
        "numeric_minimum": 100.0,
        "numeric_maximum": 5000.0,
    }

    data.update(overrides)

    return ColumnProfile(**data)


def make_profile(**overrides):
    data = {
        "factory_id": uuid4(),
        "dataset_revision_id": uuid4(),
        "row_count": 100,
        "columns": (make_column(),),
        "exact_duplicate_row_count": 5,
    }

    data.update(overrides)

    return DatasetProfile(**data)


def make_entry(**overrides):
    data = {
        "source_column": "Planned Qty",
        "canonical_column": "planned_quantity_kg",
        "method": MappingMethod.ALIAS,
        "source_unit": "kg",
        "canonical_unit": "kg",
        "requires_confirmation": True,
    }

    data.update(overrides)

    return ColumnMappingEntry(**data)


def make_proposal(**overrides):
    data = {
        "factory_id": uuid4(),
        "execution_id": uuid4(),
        "dataset_revision_id": uuid4(),
        "entries": (make_entry(),),
        "unmapped_columns": (),
    }

    data.update(overrides)

    return MappingProposal(**data)


def test_valid_column_profile():
    column = make_column()

    assert column.missing_count == 10
    assert column.unique_count == 50


def test_invalid_numeric_range():
    with pytest.raises(ValidationError):
        make_column(
            numeric_minimum=5000,
            numeric_maximum=100,
        )


def test_negative_missing_count():
    with pytest.raises(ValidationError):
        make_column(missing_count=-1)


def test_nonfinite_statistics():
    with pytest.raises(ValidationError):
        make_column(
            numeric_minimum=float("nan"),
        )


def test_valid_dataset_profile():
    profile = make_profile()

    assert profile.row_count == 100
    assert profile.column_count == 1
    assert profile.exact_duplicate_row_count == 5


def test_missing_count_exceeds_rows():
    with pytest.raises(ValidationError):
        make_profile(
            columns=(
                make_column(missing_count=101),
            ),
        )


def test_unique_count_exceeds_non_missing_rows():
    with pytest.raises(ValidationError):
        make_profile(
            columns=(
                make_column(
                    missing_count=20,
                    unique_count=90,
                ),
            ),
        )


def test_duplicate_count_exceeds_possible_rows():
    with pytest.raises(ValidationError):
        make_profile(
            exact_duplicate_row_count=100,
        )


def test_duplicate_profile_column_names():
    with pytest.raises(ValidationError):
        make_profile(
            columns=(
                make_column(),
                make_column(),
            ),
        )


def test_empty_dataset_profile():
    profile = make_profile(
        row_count=0,
        exact_duplicate_row_count=0,
        columns=(
            make_column(
                missing_count=0,
                unique_count=0,
                numeric_minimum=None,
                numeric_maximum=None,
            ),
        ),
    )

    assert profile.row_count == 0


def test_valid_mapping_proposal():
    proposal = make_proposal()

    assert proposal.requires_review


def test_mapping_without_required_review():
    proposal = make_proposal(
        entries=(
            make_entry(
                requires_confirmation=False,
            ),
        ),
    )

    assert not proposal.requires_review


def test_duplicate_source_mapping():
    with pytest.raises(ValidationError):
        make_proposal(
            entries=(
                make_entry(),
                make_entry(
                    canonical_column="another_quantity",
                ),
            ),
        )


def test_duplicate_canonical_mapping():
    with pytest.raises(ValidationError):
        make_proposal(
            entries=(
                make_entry(),
                make_entry(
                    source_column="Another Qty",
                ),
            ),
        )


def test_mapped_column_cannot_be_unmapped():
    with pytest.raises(ValidationError):
        make_proposal(
            unmapped_columns=("Planned Qty",),
        )


def test_duplicate_unmapped_columns():
    with pytest.raises(ValidationError):
        make_proposal(
            unmapped_columns=("Notes", "Notes"),
        )


def test_mapping_requires_confirmation():
    proposal = make_proposal()

    with pytest.raises(ValidationError):
        ApprovedMapping(
            proposal=proposal,
            approved_by="synthetic_reviewer",
        )


def test_valid_approved_mapping():
    proposal = make_proposal()

    mapping = ApprovedMapping(
        proposal=proposal,
        approved_by="synthetic_reviewer",
        confirmed_source_columns=("Planned Qty",),
    )

    assert mapping.proposal == proposal


def test_unknown_confirmation_rejected():
    proposal = make_proposal()

    with pytest.raises(ValidationError):
        ApprovedMapping(
            proposal=proposal,
            approved_by="synthetic_reviewer",
            confirmed_source_columns=(
                "Planned Qty",
                "Unknown Column",
            ),
        )


def test_automatic_mapping_requires_no_manual_columns():
    proposal = make_proposal(
        entries=(
            make_entry(
                requires_confirmation=False,
            ),
        ),
    )

    mapping = ApprovedMapping(
        proposal=proposal,
        approved_by="synthetic_policy_engine",
    )

    assert mapping.confirmed_source_columns == ()


def test_mapping_serialization():
    proposal = make_proposal()

    restored = MappingProposal.model_validate_json(
        proposal.model_dump_json()
    )

    assert restored == proposal


def test_profile_serialization():
    profile = make_profile()

    restored = DatasetProfile.model_validate_json(
        profile.model_dump_json()
    )

    assert restored == profile


def test_profile_is_frozen():
    profile = make_profile()

    with pytest.raises(ValidationError):
        profile.row_count = 200