"""Restricted, deterministic, vectorized row-local feature transformations."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_MUL_SAFE_FLOAT_BOUND = 9.0e18


class BinaryOperator(StrEnum):
    """Supported element-wise binary arithmetic and comparison operators."""

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    SAFE_DIV = "safe_div"
    SAFE_DIV_POSITIVE = "safe_div_positive"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class DatetimePart(StrEnum):
    """Supported row-local UTC datetime component extractions."""

    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    MONTH = "month"
    YEAR = "year"


class ColumnRef(BaseModel):
    """Reference to a declared dependency column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["column"] = "column"
    name: str = Field(min_length=1)

    def referenced_columns(self) -> frozenset[str]:
        return frozenset({self.name})

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if self.name not in frame.columns:
            raise KeyError(f"Dependency column '{self.name}' not found.")
        return frame[self.name]


class ConstantValue(BaseModel):
    """Fixed scalar constant for row-local arithmetic or comparisons."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["constant"] = "constant"
    value: int | float | bool | str

    def referenced_columns(self) -> frozenset[str]:
        return frozenset()

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if isinstance(self.value, bool):
            return pd.Series(self.value, index=frame.index, dtype="boolean")
        if isinstance(self.value, int):
            if not _INT64_MIN <= self.value <= _INT64_MAX:
                raise OverflowError("Integer constant outside signed 64-bit range.")
            return pd.Series(self.value, index=frame.index, dtype="Int64")
        if isinstance(self.value, float):
            return pd.Series(self.value, index=frame.index, dtype="Float64")
        return pd.Series(self.value, index=frame.index, dtype="string")


class BinaryExpr(BaseModel):
    """Element-wise binary operation between two row-local expressions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["binary"] = "binary"
    op: BinaryOperator
    left: RowLocalExpr
    right: RowLocalExpr

    def referenced_columns(self) -> frozenset[str]:
        return self.left.referenced_columns() | self.right.referenced_columns()

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        left_s = self.left.evaluate(frame)
        right_s = self.right.evaluate(frame)

        if self.op in {
            BinaryOperator.ADD,
            BinaryOperator.SUB,
            BinaryOperator.MUL,
            BinaryOperator.DIV,
            BinaryOperator.SAFE_DIV,
            BinaryOperator.SAFE_DIV_POSITIVE,
        }:
            l_num = _to_numeric_preserving_na(left_s)
            r_num = _to_numeric_preserving_na(right_s)

            if self.op in {BinaryOperator.ADD, BinaryOperator.SUB, BinaryOperator.MUL}:
                both_int = pd.api.types.is_integer_dtype(
                    l_num.dtype
                ) and pd.api.types.is_integer_dtype(r_num.dtype)
                if both_int:
                    return _checked_int64_arithmetic(l_num, r_num, self.op)

                with np.errstate(over="ignore", invalid="ignore"):
                    if self.op == BinaryOperator.ADD:
                        res = l_num + r_num
                    elif self.op == BinaryOperator.SUB:
                        res = l_num - r_num
                    else:
                        res = l_num * r_num
                _ensure_finite_non_null(res)
                return res

            if self.op == BinaryOperator.DIV:
                valid = l_num.notna() & r_num.notna()
                if bool((valid & (r_num == 0)).any()):
                    raise ZeroDivisionError("Division by zero in DIV expression.")
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    res = (l_num.astype("Float64") / r_num.astype("Float64")).mask(~valid, pd.NA)
                _ensure_finite_non_null(res)
                return res

            valid = l_num.notna() & r_num.notna()
            if self.op == BinaryOperator.SAFE_DIV:
                denom = r_num.astype("Float64").where(r_num != 0, pd.NA)
            else:
                denom = r_num.astype("Float64").where(r_num > 0, pd.NA)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                res = (l_num.astype("Float64") / denom).mask(~valid, pd.NA)
            _ensure_finite_non_null(res)
            return res

        if self.op == BinaryOperator.EQ:
            return (left_s == right_s).astype("boolean")
        if self.op == BinaryOperator.NE:
            return (left_s != right_s).astype("boolean")
        l_num = _to_numeric_preserving_na(left_s)
        r_num = _to_numeric_preserving_na(right_s)
        if self.op == BinaryOperator.LT:
            return (l_num < r_num).astype("boolean")
        if self.op == BinaryOperator.LE:
            return (l_num <= r_num).astype("boolean")
        if self.op == BinaryOperator.GT:
            return (l_num > r_num).astype("boolean")
        return (l_num >= r_num).astype("boolean")


class UnitScaleExpr(BaseModel):
    """Fixed linear unit conversion or scaling: value * factor + offset."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["unit_scale"] = "unit_scale"
    operand: RowLocalExpr
    factor: float
    offset: float = 0.0

    def referenced_columns(self) -> frozenset[str]:
        return self.operand.referenced_columns()

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        num = _to_numeric_preserving_na(self.operand.evaluate(frame)).astype("Float64")
        with np.errstate(over="ignore", invalid="ignore"):
            res = num * self.factor
            if self.offset != 0.0:
                res = res + self.offset
        _ensure_finite_non_null(res)
        return res


class WhereExpr(BaseModel):
    """Row-local conditional selection: if condition then when_true else when_false."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["where"] = "where"
    condition: RowLocalExpr
    when_true: RowLocalExpr
    when_false: RowLocalExpr

    def referenced_columns(self) -> frozenset[str]:
        return (
            self.condition.referenced_columns()
            | self.when_true.referenced_columns()
            | self.when_false.referenced_columns()
        )

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        cond_raw = self.condition.evaluate(frame)
        if not pd.api.types.is_bool_dtype(cond_raw.dtype):
            raise ValueError("WhereExpr condition must evaluate to a boolean Series.")
        cond_bool = cond_raw.astype("boolean")
        t_val = self.when_true.evaluate(frame)
        f_val = self.when_false.evaluate(frame)
        t_aligned, f_aligned = _align_branch_series(t_val, f_val)
        selected = t_aligned.where(cond_bool.fillna(False), f_aligned)
        return selected.mask(cond_bool.isna(), pd.NA)


class DatetimePartExpr(BaseModel):
    """Row-local UTC datetime component extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["datetime_part"] = "datetime_part"
    column: str = Field(min_length=1)
    part: DatetimePart

    def referenced_columns(self) -> frozenset[str]:
        return frozenset({self.column})

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if self.column not in frame.columns:
            raise KeyError(f"Dependency column '{self.column}' not found.")
        src = frame[self.column]
        na_mask = src.isna()
        dt = pd.to_datetime(src, format="mixed", utc=True, errors="coerce")
        if bool((dt.isna() & ~na_mask).any()):
            raise ValueError(f"Invalid datetime value in column '{self.column}'.")
        if self.part == DatetimePart.HOUR:
            return dt.dt.hour.astype("Int64")
        if self.part == DatetimePart.DAY_OF_WEEK:
            return dt.dt.dayofweek.astype("Int64")
        if self.part == DatetimePart.MONTH:
            return dt.dt.month.astype("Int64")
        return dt.dt.year.astype("Int64")


RowLocalExpr = Annotated[
    ColumnRef | ConstantValue | BinaryExpr | UnitScaleExpr | WhereExpr | DatetimePartExpr,
    Field(discriminator="kind"),
]
RowLocalTransform = RowLocalExpr

BinaryExpr.model_rebuild()
UnitScaleExpr.model_rebuild()
WhereExpr.model_rebuild()

_ALLOWED_EXPR_TYPES = frozenset(
    {
        ColumnRef,
        ConstantValue,
        BinaryExpr,
        UnitScaleExpr,
        WhereExpr,
        DatetimePartExpr,
    }
)


def validate_row_local_transform(expr: object) -> None:
    """Strictly verify that expr is an exact allowed restricted expression tree."""
    if type(expr) not in _ALLOWED_EXPR_TYPES:
        raise TypeError(f"Unsupported row-local expression node: {type(expr).__name__}.")
    if type(expr) is BinaryExpr:
        validate_row_local_transform(expr.left)
        validate_row_local_transform(expr.right)
    elif type(expr) is UnitScaleExpr:
        validate_row_local_transform(expr.operand)
    elif type(expr) is WhereExpr:
        validate_row_local_transform(expr.condition)
        validate_row_local_transform(expr.when_true)
        validate_row_local_transform(expr.when_false)


def _to_numeric_preserving_na(series: pd.Series) -> pd.Series:
    """Convert numeric or numeric-string Series while rejecting booleans."""
    if pd.api.types.is_bool_dtype(series.dtype):
        raise ValueError("Boolean series cannot be used in numeric row-local arithmetic.")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series
    return pd.to_numeric(series, errors="raise")


def _checked_int64_arithmetic(
    left: pd.Series,
    right: pd.Series,
    op: BinaryOperator,
) -> pd.Series:
    """Execute exact Int64 ADD/SUB/MUL with overflow detection (no Float64 fallback)."""
    valid = left.notna() & right.notna()
    if not bool(valid.any()):
        return pd.Series([pd.NA] * len(left), index=left.index, dtype="Int64")

    a = left[valid].to_numpy(dtype=np.int64)
    b = right[valid].to_numpy(dtype=np.int64)

    if op == BinaryOperator.ADD:
        overflow = ((b > 0) & (a > (_INT64_MAX - b))) | ((b < 0) & (a < (_INT64_MIN - b)))
        if bool(overflow.any()):
            raise OverflowError("Signed 64-bit integer addition overflow.")
        vals = a + b
    elif op == BinaryOperator.SUB:
        overflow = ((b < 0) & (a > (_INT64_MAX + b))) | ((b > 0) & (a < (_INT64_MIN + b)))
        if bool(overflow.any()):
            raise OverflowError("Signed 64-bit integer subtraction overflow.")
        vals = a - b
    else:
        suspected = (
            (a != 0)
            & (b != 0)
            & (
                (np.abs(a.astype(np.float64)) * np.abs(b.astype(np.float64)))
                >= _MUL_SAFE_FLOAT_BOUND
            )
        )
        if bool(suspected.any()):
            for ai, bi in zip(a[suspected].tolist(), b[suspected].tolist(), strict=True):
                prod = int(ai) * int(bi)
                if prod < _INT64_MIN or prod > _INT64_MAX:
                    raise OverflowError("Signed 64-bit integer multiplication overflow.")
        vals = a * b

    result = pd.Series([pd.NA] * len(left), index=left.index, dtype="Int64")
    result.loc[valid] = vals
    return result


def _ensure_finite_non_null(series: pd.Series) -> None:
    """Reject floating-point inf/-inf produced by non-null arithmetic."""
    non_null = series.dropna()
    if non_null.empty:
        return
    arr = non_null.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(arr).all()):
        raise OverflowError("Floating-point arithmetic produced non-finite value.")


def _align_branch_series(
    t_val: pd.Series,
    f_val: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align WhereExpr branch dtypes without silent lossy integer coercion."""
    if t_val.dtype == f_val.dtype:
        return t_val, f_val
    t_int = pd.api.types.is_integer_dtype(t_val.dtype)
    f_int = pd.api.types.is_integer_dtype(f_val.dtype)
    t_float = pd.api.types.is_float_dtype(t_val.dtype)
    f_float = pd.api.types.is_float_dtype(f_val.dtype)
    if (t_int and f_float) or (t_float and f_int):
        return t_val.astype("Float64"), f_val.astype("Float64")
    if t_int and f_int:
        return t_val.astype("Int64"), f_val.astype("Int64")
    if t_float and f_float:
        return t_val.astype("Float64"), f_val.astype("Float64")
    return t_val.astype("object"), f_val.astype("object")


__all__ = [
    "BinaryExpr",
    "BinaryOperator",
    "ColumnRef",
    "ConstantValue",
    "DatetimePart",
    "DatetimePartExpr",
    "RowLocalExpr",
    "RowLocalTransform",
    "UnitScaleExpr",
    "WhereExpr",
    "validate_row_local_transform",
]
