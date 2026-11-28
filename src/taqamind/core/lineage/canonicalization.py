"""Contract-driven canonicalization and typed v2 fingerprint serialization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import pandas as pd

from taqamind.core.contracts.dataset import DatasetContract
from taqamind.core.contracts.enums import DataType
from taqamind.core.contracts.feature import FeatureContract
from taqamind.core.preprocessing.missing_values import normalize_text_missing
from taqamind.core.preprocessing.type_conversion import ContractTypeConverter

FINGERPRINT_VERSION_V1 = "1.0.0"
FINGERPRINT_VERSION_V2 = "2.0.0"


class CanonicalizationError(ValueError):
    """Structured error raised when a column fails contract canonicalization."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        column: str,
        invalid_positions: tuple[int, ...] = (),
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.column = column
        self.invalid_positions = invalid_positions


@dataclass(frozen=True, slots=True)
class CanonicalizedColumn:
    """Validated canonical values and typed v2 tokens for a single column."""

    name: str
    data_type: DataType
    values: list[str | int | float | bool | None]
    v2_tokens: list[str]
    legacy_values: list[str | int | float | bool | None]


def canonicalize_column(
    series: pd.Series,
    feature: FeatureContract,
    *,
    converter: ContractTypeConverter | None = None,
) -> CanonicalizedColumn:
    """Validate and canonicalize a column using WP01 ContractTypeConverter."""
    conv = converter or ContractTypeConverter()
    normalized = normalize_text_missing(series, data_type=feature.data_type)
    result = conv.convert(normalized, data_type=feature.data_type)
    if not result.is_valid:
        raise CanonicalizationError(
            "INVALID_FINGERPRINT_INPUT",
            f"Column '{feature.name}' contains invalid values for '{feature.data_type.value}'.",
            column=feature.name,
            invalid_positions=result.invalid_positions,
        )

    raw_list = result.values.tolist()
    values: list[str | int | float | bool | None] = []
    v2_tokens: list[str] = []
    legacy_values: list[str | int | float | bool | None] = []

    dt = feature.data_type
    allowed_set = (
        set(feature.constraints.allowed_values)
        if dt == DataType.CATEGORY and feature.constraints.allowed_values is not None
        else None
    )
    bad_allowed: list[int] = []

    if dt == DataType.STRING:
        for val in raw_list:
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                s = str(val).strip()
                values.append(s)
                v2_tokens.append(f"str:{s}")
                legacy_values.append(s)
    elif dt == DataType.CATEGORY:
        for idx, val in enumerate(raw_list):
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                s = str(val).strip()
                if allowed_set is not None and s not in allowed_set:
                    bad_allowed.append(idx)
                values.append(s)
                v2_tokens.append(f"cat:{s}")
                legacy_values.append(s)
        if bad_allowed:
            raise CanonicalizationError(
                "INVALID_FINGERPRINT_INPUT",
                f"Column '{feature.name}' contains values outside allowed categories.",
                column=feature.name,
                invalid_positions=tuple(bad_allowed),
            )
    elif dt == DataType.INTEGER:
        for val in raw_list:
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                iv = int(val)
                values.append(iv)
                v2_tokens.append(f"int:{iv}")
                legacy_values.append(iv)
    elif dt == DataType.FLOAT:
        for idx, val in enumerate(raw_list):
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                fv = float(val)
                if not math.isfinite(fv):
                    raise CanonicalizationError(
                        "INVALID_FINGERPRINT_INPUT",
                        f"Column '{feature.name}' contains non-finite float values.",
                        column=feature.name,
                        invalid_positions=(idx,),
                    )
                if fv == 0.0:
                    fv = 0.0
                values.append(fv)
                v2_tokens.append(f"float:{fv.hex()}")
                legacy_values.append(round(fv, 9))
    elif dt == DataType.BOOLEAN:
        for val in raw_list:
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                bv = bool(val)
                values.append(bv)
                v2_tokens.append("bool:true" if bv else "bool:false")
                legacy_values.append(bv)
    elif dt == DataType.DATE:
        for val in raw_list:
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                dv = str(val)
                values.append(dv)
                v2_tokens.append(f"date:{dv}")
                legacy_values.append(dv)
    elif dt == DataType.DATETIME:
        for val in raw_list:
            if pd.isna(val):
                values.append(None)
                v2_tokens.append("null")
                legacy_values.append(None)
            else:
                dtv = str(val)
                values.append(dtv)
                v2_tokens.append(f"datetime:{dtv}")
                legacy_values.append(dtv)
    else:
        raise CanonicalizationError(
            "INVALID_FINGERPRINT_INPUT",
            f"Unsupported data type '{dt.value}' for canonicalization.",
            column=feature.name,
        )

    return CanonicalizedColumn(
        name=feature.name,
        data_type=dt,
        values=values,
        v2_tokens=v2_tokens,
        legacy_values=legacy_values,
    )


def hash_canonical_v2_row(
    *,
    namespace: str,
    field_pairs: list[tuple[str, str]],
    schema_provenance: list[list[str]] | None = None,
    version: str = FINGERPRINT_VERSION_V2,
) -> str:
    """Compute a deterministic typed v2 SHA-256 fingerprint for pre-sorted field pairs."""
    payload: dict[str, Any] = {
        "fields": field_pairs,
        "namespace": namespace,
        "version": version,
    }
    if schema_provenance is not None:
        payload["schema"] = schema_provenance
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def find_duplicate_canonical_business_keys(
    df: pd.DataFrame,
    contract: DatasetContract,
    *,
    converter: ContractTypeConverter | None = None,
) -> tuple[int, ...]:
    """Return zero-based duplicate row indices using canonical business key tuples."""
    if df.empty or not contract.business_keys:
        return ()
    if not all(key in df.columns for key in contract.business_keys):
        return ()

    col_map = contract.column_map
    conv = converter or ContractTypeConverter()
    canon_cols: list[CanonicalizedColumn] = []
    for key in contract.business_keys:
        try:
            canon_cols.append(canonicalize_column(df[key], col_map[key], converter=conv))
        except CanonicalizationError:
            # Invalid key tokens are reported by column validation; do not mask them as duplicates.
            return ()

    seen: set[tuple[str, ...]] = set()
    dup_indices: list[int] = []
    for idx in range(len(df)):
        tokens = tuple(col.v2_tokens[idx] for col in canon_cols)
        if "null" in tokens:
            continue
        if tokens in seen:
            dup_indices.append(idx)
        else:
            seen.add(tokens)
    return tuple(dup_indices)


__all__ = [
    "CanonicalizationError",
    "CanonicalizedColumn",
    "FINGERPRINT_VERSION_V1",
    "FINGERPRINT_VERSION_V2",
    "canonicalize_column",
    "find_duplicate_canonical_business_keys",
    "hash_canonical_v2_row",
]
