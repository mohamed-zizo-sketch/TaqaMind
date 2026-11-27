"""Shared contract-driven column validation."""

from decimal import Decimal

import numpy as np
import pandas as pd

from taqamind.core.contracts.enums import (
    DataType,
    IssueImpact,
    MissingValuePolicy,
    PipelineStage,
    ValidationSeverity,
)
from taqamind.core.contracts.feature import FeatureContract
from taqamind.core.contracts.validation import ValidationIssue
from taqamind.core.preprocessing.missing_values import (
    normalize_text_missing,
)
from taqamind.core.preprocessing.type_conversion import (
    ContractTypeConverter,
    TypeConversionResult,
)

_MAX_ROW_SAMPLES = 20


def validate_column_with_conversion(
    series: pd.Series,
    feature: FeatureContract,
    *,
    stage: PipelineStage,
    converter: ContractTypeConverter | None = None,
) -> tuple[list[ValidationIssue], TypeConversionResult | None]:
    """Validate column values without learning dataset statistics and return conversion."""

    conv = converter or ContractTypeConverter()
    issues: list[ValidationIssue] = []

    series = normalize_text_missing(
        series,
        data_type=feature.data_type,
    )

    # 1. Missing value validation.
    na_mask = series.isna()
    missing_indices = np.flatnonzero(na_mask.to_numpy())

    if len(missing_indices) > 0:
        if feature.missing_policy == MissingValuePolicy.REJECT:
            issues.append(
                ValidationIssue(
                    code="MISSING_VALUES_REJECTED",
                    stage=stage,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(f"Non-nullable column '{feature.name}' contains missing values."),
                    columns=(feature.name,),
                    affected_rows=int(len(missing_indices)),
                    row_samples=tuple(int(idx) for idx in missing_indices[:_MAX_ROW_SAMPLES]),
                    recommendation=("Populate missing values or reject incomplete rows."),
                )
            )

        elif feature.missing_policy == MissingValuePolicy.IMPUTE:
            issues.append(
                ValidationIssue(
                    code="MISSING_VALUES_PENDING_IMPUTATION",
                    stage=stage,
                    severity=ValidationSeverity.WARNING,
                    impact=IssueImpact.NONE,
                    message=(
                        f"Column '{feature.name}' contains missing values "
                        "scheduled for cleaning/imputation."
                    ),
                    columns=(feature.name,),
                    affected_rows=int(len(missing_indices)),
                    row_samples=tuple(int(idx) for idx in missing_indices[:_MAX_ROW_SAMPLES]),
                )
            )

    non_null_mask = ~na_mask

    if not bool(non_null_mask.any()):
        return issues, None

    # 2. Logical data type validation.
    conversion = conv.convert(
        series,
        data_type=feature.data_type,
    )

    if not conversion.is_valid:
        issues.append(
            ValidationIssue(
                code="INVALID_DATA_TYPE",
                stage=stage,
                severity=ValidationSeverity.ERROR,
                impact=IssueImpact.BLOCKING,
                message=(
                    f"Column '{feature.name}' contains values "
                    f"incompatible with '{feature.data_type.value}'."
                ),
                columns=(feature.name,),
                affected_rows=len(conversion.invalid_positions),
                row_samples=(conversion.invalid_positions[:_MAX_ROW_SAMPLES]),
            )
        )

    constraints = feature.constraints

    # 3. Numeric range validation.
    if feature.data_type in {DataType.INTEGER, DataType.FLOAT} and (
        constraints.minimum is not None or constraints.maximum is not None
    ):
        min_dec = Decimal(str(constraints.minimum)) if constraints.minimum is not None else None

        max_dec = Decimal(str(constraints.maximum)) if constraints.maximum is not None else None

        range_bad_indices: list[int] = []

        for idx, val in enumerate(conversion.values):
            if pd.isna(val):
                continue

            dec_val = (
                Decimal(int(val)) if feature.data_type == DataType.INTEGER else Decimal(str(val))
            )

            if min_dec is not None:
                if (constraints.minimum_inclusive and dec_val < min_dec) or (
                    not constraints.minimum_inclusive and dec_val <= min_dec
                ):
                    range_bad_indices.append(idx)
                    continue

            if max_dec is not None:
                if (constraints.maximum_inclusive and dec_val > max_dec) or (
                    not constraints.maximum_inclusive and dec_val >= max_dec
                ):
                    range_bad_indices.append(idx)

        if range_bad_indices:
            issues.append(
                ValidationIssue(
                    code="CONSTRAINT_RANGE_VIOLATION",
                    stage=stage,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(f"Column '{feature.name}' violates declared numeric bounds."),
                    columns=(feature.name,),
                    affected_rows=len(range_bad_indices),
                    row_samples=tuple(range_bad_indices[:_MAX_ROW_SAMPLES]),
                )
            )

    # 4. Allowed categorical values validation.
    if constraints.allowed_values is not None:
        allowed = set(constraints.allowed_values)

        str_values = series[non_null_mask].astype(str).str.strip()

        invalid_mask = ~str_values.isin(allowed)

        bad_positions = np.flatnonzero(non_null_mask.to_numpy())[invalid_mask.to_numpy()]

        if len(bad_positions) > 0:
            issues.append(
                ValidationIssue(
                    code="CONSTRAINT_ALLOWED_VALUES_VIOLATION",
                    stage=stage,
                    severity=ValidationSeverity.ERROR,
                    impact=IssueImpact.BLOCKING,
                    message=(f"Column '{feature.name}' contains values outside allowed set."),
                    columns=(feature.name,),
                    affected_rows=int(len(bad_positions)),
                    row_samples=tuple(int(idx) for idx in bad_positions[:_MAX_ROW_SAMPLES]),
                )
            )

    return issues, conversion


def validate_column(
    series: pd.Series,
    feature: FeatureContract,
    *,
    stage: PipelineStage,
    converter: ContractTypeConverter | None = None,
) -> list[ValidationIssue]:
    """Validate column values without learning dataset statistics."""

    issues, _ = validate_column_with_conversion(
        series,
        feature,
        stage=stage,
        converter=converter,
    )
    return issues
