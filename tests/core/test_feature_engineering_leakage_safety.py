"""Regression and invariance tests for WP01.8 leakage-safe feature engineering (REV-025)."""

import time
from pathlib import Path

import pandas as pd
import pytest
from test_cleaning_feature_engineering_and_preprocessing import (
    make_contract_and_context,
    map_csv,
)

from taqamind.core.contracts import DatasetContract, DataType, MissingValuePolicy
from taqamind.core.feature_engineering import (
    BinaryExpr,
    BinaryOperator,
    ColumnRef,
    ConstantValue,
    DerivedFeatureSpec,
    DeterministicFeatureEngineer,
    UnitScaleExpr,
    WhereExpr,
)
from taqamind.core.interfaces import ExecutionStatus
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.registry import LocalDatasetArtifactStorage


def _run_engineer(
    tmp_path: Path,
    csv_bytes: bytes,
    specs: tuple[DerivedFeatureSpec, ...],
    *,
    contract: DatasetContract | None = None,
    allow_unverified_callbacks: bool = False,
):
    default_contract, context = make_contract_and_context()
    active_contract = contract or default_contract
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(storage, context, active_contract, csv_bytes)
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=active_contract)
        .output_artifact
    )
    assert cleaned is not None
    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=specs,
        allow_unverified_callbacks=allow_unverified_callbacks,
    ).engineer(context=context, artifact=cleaned, contract=active_contract)
    return result, storage, context


@pytest.mark.parametrize(
    "name, fn",
    [
        ("global_mean", lambda f: f["machine_speed"] / f["machine_speed"].mean()),
        ("global_median", lambda f: f["machine_speed"] - f["machine_speed"].median()),
        ("cross_row_rank", lambda f: f["machine_speed"].rank()),
        ("cross_row_shift", lambda f: f["machine_speed"].shift(1).fillna(0.0)),
    ],
)
def test_default_feature_engineer_blocks_unverified_callbacks(tmp_path: Path, name: str, fn):
    csv_bytes = b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\nB2,150,450,3\n"
    spec = DerivedFeatureSpec(feature_name="speed_per_kg", compute=fn)
    res, _, _ = _run_engineer(tmp_path, csv_bytes, (spec,))
    assert res.status == ExecutionStatus.BLOCKED, name
    assert res.output_artifact is None
    assert res.report is not None
    assert any(i.code == "UNVERIFIED_CALLBACK_BLOCKED" for i in res.report.issues)


def test_compatibility_mode_emits_explicit_warning(tmp_path: Path):
    csv_bytes = b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n"
    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda f: f["machine_speed"] / f["planned_qty"],
    )
    res, _, _ = _run_engineer(
        tmp_path,
        csv_bytes,
        (spec,),
        allow_unverified_callbacks=True,
    )
    assert res.status == ExecutionStatus.SUCCEEDED
    assert res.report is not None
    assert any(i.code == "UNVERIFIED_CALLBACK_COMPATIBILITY_MODE" for i in res.report.issues)


def test_restricted_transform_blocks_undeclared_column_reference(tmp_path: Path):
    csv_bytes = b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n"
    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        transform=BinaryExpr(
            op=BinaryOperator.DIV,
            left=ColumnRef(name="waste_qty"),
            right=ColumnRef(name="planned_qty"),
        ),
    )
    res, _, _ = _run_engineer(tmp_path, csv_bytes, (spec,))
    assert res.status == ExecutionStatus.BLOCKED
    assert res.output_artifact is None
    assert res.report is not None
    assert any(i.code == "UNDECLARED_TRANSFORM_DEPENDENCY" for i in res.report.issues)


def test_row_local_transform_invariance_and_row_independence(tmp_path: Path):
    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        transform=WhereExpr(
            condition=BinaryExpr(
                op=BinaryOperator.GT,
                left=ColumnRef(name="planned_qty"),
                right=ConstantValue(value=0.0),
            ),
            when_true=UnitScaleExpr(
                operand=BinaryExpr(
                    op=BinaryOperator.SAFE_DIV_POSITIVE,
                    left=ColumnRef(name="machine_speed"),
                    right=ColumnRef(name="planned_qty"),
                ),
                factor=1.0,
            ),
            when_false=ConstantValue(value=0.0),
        ),
    )

    # Single record evaluated independently
    res_single, st1, ctx1 = _run_engineer(
        tmp_path / "single",
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
        (spec,),
    )
    # Same record alongside other records with different values
    res_multi, st2, ctx2 = _run_engineer(
        tmp_path / "multi",
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\nB2,200,1000,5\nB3,50,100,1\n",
        (spec,),
    )
    # Reversed row order
    res_rev, st3, ctx3 = _run_engineer(
        tmp_path / "rev",
        b"batch_id,planned_qty,machine_speed,waste_qty\nB3,50,100,1\nB2,200,1000,5\nB1,100,300,2\n",
        (spec,),
    )
    assert res_single.status == ExecutionStatus.SUCCEEDED
    assert res_multi.status == ExecutionStatus.SUCCEEDED
    assert res_rev.status == ExecutionStatus.SUCCEEDED

    with st1.open_verified(
        factory_id=ctx1.factory_id, reference=res_single.output_artifact.reference
    ) as s:
        df1 = pd.read_csv(s)
    with st2.open_verified(
        factory_id=ctx2.factory_id, reference=res_multi.output_artifact.reference
    ) as s:
        df2 = pd.read_csv(s)
    with st3.open_verified(
        factory_id=ctx3.factory_id, reference=res_rev.output_artifact.reference
    ) as s:
        df3 = pd.read_csv(s)

    b1_single = df1.loc[df1["batch_id"] == "B1", "speed_per_kg"].iloc[0]
    b1_multi = df2.loc[df2["batch_id"] == "B1", "speed_per_kg"].iloc[0]
    b1_rev = df3.loc[df3["batch_id"] == "B1", "speed_per_kg"].iloc[0]
    assert b1_single == pytest.approx(b1_multi)
    assert b1_single == pytest.approx(b1_rev)


def test_row_local_transform_preserves_nullable_and_exact_int64(tmp_path: Path):
    contract, _ = make_contract_and_context()
    features = tuple(
        f.model_copy(update={"data_type": DataType.INTEGER})
        if f.name in {"planned_qty", "machine_speed", "speed_per_kg"}
        else f
        for f in contract.features
    )
    int_contract = DatasetContract.model_validate({**contract.model_dump(), "features": features})

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        transform=BinaryExpr(
            op=BinaryOperator.ADD,
            left=ColumnRef(name="machine_speed"),
            right=ColumnRef(name="planned_qty"),
        ),
    )
    csv_bytes = b"batch_id,planned_qty,machine_speed,waste_qty\nB1,1,9007199254740992,2\nB2,10,,2\n"
    res, st, ctx = _run_engineer(tmp_path, csv_bytes, (spec,), contract=int_contract)
    assert res.status == ExecutionStatus.SUCCEEDED
    assert res.output_artifact is not None
    with st.open_verified(factory_id=ctx.factory_id, reference=res.output_artifact.reference) as s:
        out = pd.read_csv(s, dtype={"speed_per_kg": "string"})
    assert out.loc[0, "speed_per_kg"] == "9007199254740993"
    assert pd.isna(out.loc[1, "speed_per_kg"])


def test_row_local_transform_250k_rows_performance():
    n = 250_000
    df = pd.DataFrame(
        {
            "machine_speed": pd.Series([300.0] * n, dtype="Float64"),
            "planned_qty": pd.Series([100.0] * n, dtype="Float64"),
        }
    )
    expr = BinaryExpr(
        op=BinaryOperator.SAFE_DIV_POSITIVE,
        left=ColumnRef(name="machine_speed"),
        right=ColumnRef(name="planned_qty"),
    )
    t0 = time.perf_counter()
    out = expr.evaluate(df)
    elapsed = time.perf_counter() - t0
    assert len(out) == n
    assert float(out.iloc[0]) == pytest.approx(3.0)
    assert elapsed < 2.0


def _int_contract_with_allow_missing():
    contract, _ = make_contract_and_context()
    features = tuple(
        f.model_copy(
            update={
                "data_type": DataType.INTEGER,
                "nullable": True,
                "missing_policy": MissingValuePolicy.ALLOW,
            }
        )
        if f.name in {"planned_qty", "machine_speed", "speed_per_kg"}
        else f
        for f in contract.features
    )
    return DatasetContract.model_validate({**contract.model_dump(), "features": features})


def _float_contract_with_allow_missing():
    contract, _ = make_contract_and_context()
    features = tuple(
        f.model_copy(
            update={
                "data_type": DataType.FLOAT,
                "nullable": True,
                "missing_policy": MissingValuePolicy.ALLOW,
            }
        )
        if f.name in {"planned_qty", "machine_speed", "speed_per_kg"}
        else f
        for f in contract.features
    )
    return DatasetContract.model_validate({**contract.model_dump(), "features": features})


@pytest.mark.parametrize(
    "csv_row, expr",
    [
        (
            b"B1,1,9223372036854775807,2\n",
            BinaryExpr(
                op=BinaryOperator.ADD,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        (
            b"B1,1,-9223372036854775808,2\n",
            BinaryExpr(
                op=BinaryOperator.SUB,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        (
            b"B1,1099511627776,1099511627776,2\n",
            BinaryExpr(
                op=BinaryOperator.MUL,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        (
            b"B1,-1,-9223372036854775808,2\n",
            BinaryExpr(
                op=BinaryOperator.MUL,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        # Nested overflow that would wrap back to INT64_MAX if unchecked!
        (
            b"B1,1,9223372036854775807,2\n",
            BinaryExpr(
                op=BinaryOperator.SUB,
                left=BinaryExpr(
                    op=BinaryOperator.ADD,
                    left=ColumnRef(name="machine_speed"),
                    right=ColumnRef(name="planned_qty"),
                ),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
    ],
)
def test_integer_overflow_is_blocked_without_wraparound(tmp_path: Path, csv_row: bytes, expr):
    from taqamind.core.contracts import MissingValuePolicy  # noqa: F401

    int_contract = _int_contract_with_allow_missing()
    spec = DerivedFeatureSpec(feature_name="speed_per_kg", transform=expr)
    csv_bytes = b"batch_id,planned_qty,machine_speed,waste_qty\n" + csv_row
    res, _, _ = _run_engineer(tmp_path, csv_bytes, (spec,), contract=int_contract)
    assert res.status == ExecutionStatus.BLOCKED
    assert res.output_artifact is None
    assert res.report is not None
    assert any(i.code == "DERIVED_FEATURE_COMPUTATION_FAILED" for i in res.report.issues)


def test_mixed_int_float_and_exact_large_int_mul(tmp_path: Path):
    int_contract = _int_contract_with_allow_missing()
    spec_mul = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        transform=BinaryExpr(
            op=BinaryOperator.MUL,
            left=ColumnRef(name="machine_speed"),
            right=ConstantValue(value=2),
        ),
    )
    res, st, ctx = _run_engineer(
        tmp_path / "exact_mul",
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,1,4503599627370497,2\n",
        (spec_mul,),
        contract=int_contract,
    )
    assert res.status == ExecutionStatus.SUCCEEDED
    with st.open_verified(factory_id=ctx.factory_id, reference=res.output_artifact.reference) as s:
        out = pd.read_csv(s, dtype={"speed_per_kg": "string"})
    assert out.loc[0, "speed_per_kg"] == "9007199254740994"

    # Mixed Int64 + Float64
    df_mixed = pd.DataFrame(
        {
            "a": pd.Series([10, pd.NA], dtype="Int64"),
            "b": pd.Series([2.5, 1.0], dtype="Float64"),
        }
    )
    mixed_res = BinaryExpr(
        op=BinaryOperator.ADD, left=ColumnRef(name="a"), right=ColumnRef(name="b")
    ).evaluate(df_mixed)
    assert float(mixed_res.iloc[0]) == pytest.approx(12.5)
    assert pd.isna(mixed_res.iloc[1])


@pytest.mark.parametrize(
    "csv_row",
    [
        b"B1,0,10,2\n",
        b"B1,0,0,2\n",
    ],
)
def test_ordinary_div_by_zero_blocked_even_with_allow_missing(tmp_path: Path, csv_row: bytes):
    float_contract = _float_contract_with_allow_missing()
    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        transform=BinaryExpr(
            op=BinaryOperator.DIV,
            left=ColumnRef(name="machine_speed"),
            right=ColumnRef(name="planned_qty"),
        ),
    )
    res, _, _ = _run_engineer(
        tmp_path,
        b"batch_id,planned_qty,machine_speed,waste_qty\n" + csv_row,
        (spec,),
        contract=float_contract,
    )
    assert res.status == ExecutionStatus.BLOCKED
    assert res.output_artifact is None
    assert res.report is not None
    assert any(i.code == "DERIVED_FEATURE_COMPUTATION_FAILED" for i in res.report.issues)


def test_float_overflow_and_nested_nan_masking_blocked(tmp_path: Path):
    float_contract = _float_contract_with_allow_missing()
    overflow_expr = BinaryExpr(
        op=BinaryOperator.MUL,
        left=ColumnRef(name="machine_speed"),
        right=ConstantValue(value=10.0),
    )
    nested_inf_minus_inf = BinaryExpr(
        op=BinaryOperator.SUB,
        left=overflow_expr,
        right=overflow_expr,
    )
    spec = DerivedFeatureSpec(feature_name="speed_per_kg", transform=nested_inf_minus_inf)
    res, _, _ = _run_engineer(
        tmp_path,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,1,1e308,2\n",
        (spec,),
        contract=float_contract,
    )
    assert res.status == ExecutionStatus.BLOCKED
    assert res.report is not None
    assert any(i.code == "DERIVED_FEATURE_COMPUTATION_FAILED" for i in res.report.issues)


def test_safe_div_and_safe_div_positive_semantics():
    df = pd.DataFrame(
        {
            "num": pd.Series([10.0, 10.0, 10.0, pd.NA], dtype="Float64"),
            "den": pd.Series([0.0, -2.0, 2.0, 2.0], dtype="Float64"),
        }
    )
    safe_div = BinaryExpr(
        op=BinaryOperator.SAFE_DIV, left=ColumnRef(name="num"), right=ColumnRef(name="den")
    ).evaluate(df)
    assert pd.isna(safe_div.iloc[0])
    assert float(safe_div.iloc[1]) == pytest.approx(-5.0)
    assert float(safe_div.iloc[2]) == pytest.approx(5.0)
    assert pd.isna(safe_div.iloc[3])

    safe_pos = BinaryExpr(
        op=BinaryOperator.SAFE_DIV_POSITIVE, left=ColumnRef(name="num"), right=ColumnRef(name="den")
    ).evaluate(df)
    assert pd.isna(safe_pos.iloc[0])
    assert pd.isna(safe_pos.iloc[1])
    assert float(safe_pos.iloc[2]) == pytest.approx(5.0)
    assert pd.isna(safe_pos.iloc[3])


def test_datetime_part_preserves_na_and_rejects_invalid_strings():
    from taqamind.core.feature_engineering import DatetimePart, DatetimePartExpr

    expr = DatetimePartExpr(column="ts", part=DatetimePart.HOUR)
    valid_df = pd.DataFrame({"ts": pd.Series(["2026-11-27T14:30:00Z", pd.NA], dtype="string")})
    out = expr.evaluate(valid_df)
    assert int(out.iloc[0]) == 14
    assert pd.isna(out.iloc[1])

    bad_df = pd.DataFrame(
        {"ts": pd.Series(["not-a-datetime", "2026-11-27T14:30:00Z"], dtype="string")}
    )
    with pytest.raises(ValueError, match="Invalid datetime"):
        expr.evaluate(bad_df)


def test_where_expr_missing_condition_yields_na_and_aligns_branch_dtypes():
    df = pd.DataFrame(
        {
            "x": pd.Series([10, pd.NA, -5], dtype="Int64"),
            "y": pd.Series([1.5, 2.5, 3.5], dtype="Float64"),
        }
    )
    expr = WhereExpr(
        condition=BinaryExpr(
            op=BinaryOperator.GT, left=ColumnRef(name="x"), right=ConstantValue(value=0)
        ),
        when_true=ColumnRef(name="x"),
        when_false=ColumnRef(name="y"),
    )
    out = expr.evaluate(df)
    assert float(out.iloc[0]) == pytest.approx(10.0)
    assert pd.isna(out.iloc[1])
    assert float(out.iloc[2]) == pytest.approx(3.5)


def test_unsupported_custom_transform_object_rejected():
    class _FakeColumnRef(ColumnRef):
        def evaluate(self, frame: pd.DataFrame) -> pd.Series:
            return frame[self.name] / frame[self.name].mean()

    with pytest.raises(TypeError, match="Unsupported row-local expression node"):
        DerivedFeatureSpec(
            feature_name="speed_per_kg", transform=_FakeColumnRef(name="machine_speed")
        )


def test_row_local_int64_250k_rows_performance():
    n = 250_000
    df = pd.DataFrame(
        {
            "a": pd.Series([100] * n, dtype="Int64"),
            "b": pd.Series([25] * n, dtype="Int64"),
        }
    )
    expr = BinaryExpr(op=BinaryOperator.ADD, left=ColumnRef(name="a"), right=ColumnRef(name="b"))
    t0 = time.perf_counter()
    out = expr.evaluate(df)
    elapsed = time.perf_counter() - t0
    assert len(out) == n
    assert int(out.iloc[0]) == 125
    assert elapsed < 2.0
