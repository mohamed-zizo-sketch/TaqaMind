"""Resource policies for safe CSV processing."""

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class CsvResourcePolicy:
    """Configurable resource limits for CSV input."""

    max_bytes: int = 50 * 1024 * 1024
    max_rows: int = 250_000
    max_columns: int = 500
    max_cells: int = 5_000_000

    def __post_init__(self) -> None:
        limits = {
            "max_bytes": self.max_bytes,
            "max_rows": self.max_rows,
            "max_columns": self.max_columns,
            "max_cells": self.max_cells,
        }

        for name, value in limits.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


DEFAULT_RAW_CSV_POLICY = CsvResourcePolicy()


# Internal artifacts can expand during mapping and feature engineering.
DEFAULT_INTERNAL_CSV_POLICY = CsvResourcePolicy(
    max_bytes=100 * 1024 * 1024,
    max_rows=250_000,
    max_columns=1_000,
    max_cells=10_000_000,
)


def resolve_internal_csv_policy(
    resource_policy: CsvResourcePolicy | None = None,
    *,
    max_rows: int | None = None,
) -> CsvResourcePolicy:
    """Resolve legacy row limits without silently overriding a supplied policy."""
    if resource_policy is not None and max_rows is not None:
        raise ValueError("Use resource_policy or max_rows, not both.")
    policy = resource_policy if resource_policy is not None else DEFAULT_INTERNAL_CSV_POLICY
    return replace(policy, max_rows=max_rows) if max_rows is not None else policy
