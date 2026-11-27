"""Regression tests for derived feature output validation."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from test_cleaning_feature_engineering_and_preprocessing import (
    make_contract_and_context,
    map_csv,
)

from taqamind.core.contracts import (
    DatasetContract,
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureConstraints,
    FeatureContract,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
)
from taqamind.core.feature_engineering import (
    BinaryExpr,
    BinaryOperator,
    ColumnRef,
    ConstantValue,
    DerivedFeatureSpec,
    DeterministicFeatureEngineer,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    ExecutionStatus,
)
from taqamind.core.preprocessing import ContractDataCleaner
from taqamind.core.registry import LocalDatasetArtifactStorage


@pytest.mark.parametrize(
    "value",
    ["abc", float("inf"), float("-inf")],
)
def test_invalid_derived_output_is_blocked(
    tmp_path: Path,
    value,
):
    contract, context = make_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda frame: pd.Series(
            [value] * len(frame),
            index=frame.index,
        ),
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=(spec,),
        allow_unverified_callbacks=True,
    ).engineer(
        context=context,
        artifact=cleaned,
        contract=contract,
    )

    assert result.status == ExecutionStatus.BLOCKED
    assert result.output_artifact is None
    assert result.report is not None

    assert any(issue.code == "INVALID_DATA_TYPE" for issue in result.report.issues)


@pytest.mark.parametrize("with_spec", [False, True])
def test_precomputed_derived_feature(
    tmp_path: Path,
    with_spec: bool,
):
    contract, context = make_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    payload = b"batch_id,planned_qty,machine_speed,waste_qty,speed_per_kg\nB1,100,300,2,999\n"

    reference = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )

    storage.put(
        factory_id=context.factory_id,
        reference=reference,
        stream=BytesIO(payload),
    )

    metadata = cleaned.metadata.model_copy(
        update={
            "columns": (
                *cleaned.metadata.columns,
                DatasetColumnSchema(
                    name="speed_per_kg",
                    physical_dtype="string",
                ),
            ),
        }
    )

    artifact = DatasetArtifact(
        metadata=metadata,
        reference=reference,
    )

    specs = (
        (
            DerivedFeatureSpec(
                feature_name="speed_per_kg",
                transform=BinaryExpr(
                    op=BinaryOperator.DIV,
                    left=ColumnRef(name="machine_speed"),
                    right=ColumnRef(name="planned_qty"),
                ),
            ),
        )
        if with_spec
        else ()
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=specs,
    ).engineer(
        context=context,
        artifact=artifact,
        contract=contract,
    )

    if not with_spec:
        assert result.status == ExecutionStatus.BLOCKED
        assert result.output_artifact is None
        assert result.report is not None

        assert any(issue.code == "UNVERIFIED_PRECOMPUTED_FEATURE" for issue in result.report.issues)

    else:
        assert result.status == ExecutionStatus.SUCCEEDED
        assert result.output_artifact is not None

        with storage.open_verified(
            factory_id=context.factory_id,
            reference=result.output_artifact.reference,
        ) as stream:
            output = pd.read_csv(stream)

        assert output.loc[0, "speed_per_kg"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "error_type",
    [ValueError, ZeroDivisionError, RuntimeError, KeyError],
)
def test_callback_exception_returns_structured_failure(
    tmp_path: Path,
    error_type,
):
    contract, context = make_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    def failing_compute(frame: pd.DataFrame) -> pd.Series:
        raise error_type("SENSITIVE_INTERNAL_DETAILS")

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=failing_compute,
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=(spec,),
        allow_unverified_callbacks=True,
    ).engineer(
        context=context,
        artifact=cleaned,
        contract=contract,
    )

    assert result.status == ExecutionStatus.BLOCKED
    assert result.output_artifact is None
    assert result.report is not None

    matching_issues = [
        issue
        for issue in result.report.issues
        if issue.code == "DERIVED_FEATURE_COMPUTATION_FAILED"
    ]

    assert len(matching_issues) == 1

    issue = matching_issues[0]

    assert issue.columns == ("speed_per_kg",)
    assert error_type.__name__ in issue.message
    assert "SENSITIVE_INTERNAL_DETAILS" not in issue.message


@pytest.mark.parametrize(
    "value, expected_status",
    [
        (0.0, ExecutionStatus.SUCCEEDED),
        (5.0, ExecutionStatus.SUCCEEDED),
        (10.0, ExecutionStatus.SUCCEEDED),
        (-1.0, ExecutionStatus.BLOCKED),
        (10.1, ExecutionStatus.BLOCKED),
    ],
)
def test_derived_feature_numeric_constraints(
    tmp_path: Path,
    value,
    expected_status,
):
    contract, context = make_contract_and_context()

    features = tuple(
        feature.model_copy(
            update={
                "constraints": FeatureConstraints(
                    minimum=0.0,
                    maximum=10.0,
                )
            }
        )
        if feature.name == "speed_per_kg"
        else feature
        for feature in contract.features
    )

    contract = DatasetContract.model_validate(
        {
            **contract.model_dump(),
            "features": features,
        }
    )

    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda frame: pd.Series(
            [value] * len(frame),
            index=frame.index,
        ),
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=(spec,),
        allow_unverified_callbacks=True,
    ).engineer(
        context=context,
        artifact=cleaned,
        contract=contract,
    )

    assert result.status == expected_status

    if expected_status == ExecutionStatus.BLOCKED:
        assert result.output_artifact is None
        assert result.report is not None

        assert any(issue.code == "CONSTRAINT_RANGE_VIOLATION" for issue in result.report.issues)
    else:
        assert result.output_artifact is not None


@pytest.mark.parametrize(
    "policy, expected_status",
    [
        (MissingValuePolicy.REJECT, ExecutionStatus.BLOCKED),
        (MissingValuePolicy.ALLOW, ExecutionStatus.SUCCEEDED),
        (MissingValuePolicy.IMPUTE, ExecutionStatus.SUCCEEDED),
    ],
)
def test_derived_feature_missing_policy(
    tmp_path: Path,
    policy,
    expected_status,
):
    contract, context = make_contract_and_context()

    features = tuple(
        feature.model_copy(
            update={
                "nullable": policy != MissingValuePolicy.REJECT,
                "missing_policy": policy,
            }
        )
        if feature.name == "speed_per_kg"
        else feature
        for feature in contract.features
    )

    contract = DatasetContract.model_validate(
        {
            **contract.model_dump(),
            "features": features,
        }
    )

    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda frame: pd.Series(
            pd.NA,
            index=frame.index,
            dtype="Float64",
        ),
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=(spec,),
        allow_unverified_callbacks=True,
    ).engineer(
        context=context,
        artifact=cleaned,
        contract=contract,
    )

    assert result.status == expected_status
    assert result.report is not None

    codes = {issue.code for issue in result.report.issues}

    if policy == MissingValuePolicy.REJECT:
        assert result.output_artifact is None
        assert "MISSING_VALUES_REJECTED" in codes

    else:
        assert result.output_artifact is not None

        with storage.open_verified(
            factory_id=context.factory_id,
            reference=result.output_artifact.reference,
        ) as stream:
            output = pd.read_csv(stream)

        assert pd.isna(output.loc[0, "speed_per_kg"])

        if policy == MissingValuePolicy.IMPUTE:
            assert "MISSING_VALUES_PENDING_IMPUTATION" in codes


@pytest.mark.parametrize(
    "value, expected_status",
    [
        ("9007199254740993", ExecutionStatus.SUCCEEDED),
        ("9223372036854775807", ExecutionStatus.SUCCEEDED),
        ("9223372036854775808", ExecutionStatus.BLOCKED),
        ("12.5", ExecutionStatus.BLOCKED),
    ],
)
def test_derived_feature_integer_precision(
    tmp_path: Path,
    value,
    expected_status,
):
    contract, context = make_contract_and_context()

    features = tuple(
        feature.model_copy(update={"data_type": DataType.INTEGER})
        if feature.name == "speed_per_kg"
        else feature
        for feature in contract.features
    )

    contract = DatasetContract.model_validate(
        {
            **contract.model_dump(),
            "features": features,
        }
    )

    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )

    cleaned = (
        ContractDataCleaner(storage)
        .clean(
            context=context,
            artifact=mapped,
            contract=contract,
        )
        .output_artifact
    )

    assert cleaned is not None

    spec = DerivedFeatureSpec(
        feature_name="speed_per_kg",
        compute=lambda frame: pd.Series(
            [value] * len(frame),
            index=frame.index,
            dtype="string",
        ),
    )

    result = DeterministicFeatureEngineer(
        storage,
        derived_specs=(spec,),
        allow_unverified_callbacks=True,
    ).engineer(
        context=context,
        artifact=cleaned,
        contract=contract,
    )

    assert result.status == expected_status

    if expected_status == ExecutionStatus.BLOCKED:
        assert result.output_artifact is None
        assert result.report is not None

        assert any(issue.code == "INVALID_DATA_TYPE" for issue in result.report.issues)
    else:
        assert result.output_artifact is not None

        with storage.open_verified(
            factory_id=context.factory_id,
            reference=result.output_artifact.reference,
        ) as stream:
            output = pd.read_csv(
                stream,
                dtype={"speed_per_kg": "string"},
            )

        assert output.loc[0, "speed_per_kg"] == value


def _chained_contract_and_context():
    contract, context = make_contract_and_context()
    extra_feature = FeatureContract(
        name="double_speed_per_kg",
        display_name="Double Speed Per Kg",
        description="Twice speed_per_kg.",
        data_type=DataType.FLOAT,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        derived_from=("speed_per_kg",),
    )
    contract = DatasetContract.model_validate(
        {
            **contract.model_dump(),
            "features": (*contract.features, extra_feature),
        }
    )
    return contract, context


def test_valid_chained_derived_dependencies_and_stale_precomputed_discarded(tmp_path: Path):
    contract, context = _chained_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    # Inject stale precomputed values for speed_per_kg (invalid) and double_speed_per_kg.
    payload = (
        b"batch_id,planned_qty,machine_speed,waste_qty,speed_per_kg,double_speed_per_kg\n"
        b"B1,100,300,2,INVALID_STALE,9999\n"
    )
    ref = ArtifactReference(
        factory_id=context.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=context.factory_id, reference=ref, stream=BytesIO(payload))
    meta = cleaned.metadata.model_copy(
        update={
            "columns": (
                *cleaned.metadata.columns,
                DatasetColumnSchema(name="speed_per_kg", physical_dtype="string"),
                DatasetColumnSchema(name="double_speed_per_kg", physical_dtype="string"),
            )
        }
    )
    artifact = DatasetArtifact(metadata=meta, reference=ref)

    specs = (
        DerivedFeatureSpec(
            feature_name="speed_per_kg",
            transform=BinaryExpr(
                op=BinaryOperator.DIV,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        DerivedFeatureSpec(
            feature_name="double_speed_per_kg",
            transform=BinaryExpr(
                op=BinaryOperator.MUL,
                left=ColumnRef(name="speed_per_kg"),
                right=ConstantValue(value=2.0),
            ),
        ),
    )
    res = DeterministicFeatureEngineer(storage, derived_specs=specs).engineer(
        context=context, artifact=artifact, contract=contract
    )
    assert res.status == ExecutionStatus.SUCCEEDED
    assert res.output_artifact is not None
    with storage.open_verified(
        factory_id=context.factory_id, reference=res.output_artifact.reference
    ) as s:
        out = pd.read_csv(s)
    assert out.loc[0, "speed_per_kg"] == pytest.approx(3.0)
    assert out.loc[0, "double_speed_per_kg"] == pytest.approx(6.0)


def test_invalid_intermediate_dependency_blocks_downstream_execution(tmp_path: Path):
    contract, context = _chained_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    downstream_calls = 0

    def _downstream(frame: pd.DataFrame) -> pd.Series:
        nonlocal downstream_calls
        downstream_calls += 1
        return frame["speed_per_kg"] * 2.0

    specs = (
        DerivedFeatureSpec(
            feature_name="speed_per_kg",
            compute=lambda frame: pd.Series(["not_a_float"], index=frame.index),
        ),
        DerivedFeatureSpec(
            feature_name="double_speed_per_kg",
            compute=_downstream,
        ),
    )
    res = DeterministicFeatureEngineer(
        storage, derived_specs=specs, allow_unverified_callbacks=True
    ).engineer(context=context, artifact=cleaned, contract=contract)

    assert res.status == ExecutionStatus.BLOCKED
    assert res.output_artifact is None
    assert downstream_calls == 0


def test_callback_wrong_row_count_wrong_index_and_non_scalar_values(tmp_path: Path):
    contract, context = make_contract_and_context()
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    # Wrong row count
    res_len = DeterministicFeatureEngineer(
        storage,
        derived_specs=(DerivedFeatureSpec("speed_per_kg", lambda f: pd.Series([1.0, 2.0])),),
        allow_unverified_callbacks=True,
    ).engineer(context=context, artifact=cleaned, contract=contract)
    assert res_len.status == ExecutionStatus.BLOCKED
    assert res_len.report is not None
    assert any(i.code == "DERIVED_FEATURE_LENGTH_MISMATCH" for i in res_len.report.issues)

    # Wrong index
    res_idx = DeterministicFeatureEngineer(
        storage,
        derived_specs=(DerivedFeatureSpec("speed_per_kg", lambda f: pd.Series([1.0], index=[99])),),
        allow_unverified_callbacks=True,
    ).engineer(context=context, artifact=cleaned, contract=contract)
    assert res_idx.status == ExecutionStatus.BLOCKED
    assert res_idx.report is not None
    assert any(i.code == "DERIVED_FEATURE_LENGTH_MISMATCH" for i in res_idx.report.issues)

    # Non-scalar values inside Series
    res_nonscalar = DeterministicFeatureEngineer(
        storage,
        derived_specs=(
            DerivedFeatureSpec("speed_per_kg", lambda f: pd.Series([[1, 2]], index=f.index)),
        ),
        allow_unverified_callbacks=True,
    ).engineer(context=context, artifact=cleaned, contract=contract)
    assert res_nonscalar.status == ExecutionStatus.BLOCKED
    assert res_nonscalar.output_artifact is None
    assert res_nonscalar.report is not None
    assert any(i.code == "INVALID_DATA_TYPE" for i in res_nonscalar.report.issues)


@pytest.mark.parametrize(
    "value, expected_status",
    [
        (0.0, ExecutionStatus.BLOCKED),
        (5.0, ExecutionStatus.SUCCEEDED),
        (10.0, ExecutionStatus.BLOCKED),
    ],
)
def test_derived_feature_exclusive_numeric_bounds(tmp_path: Path, value: float, expected_status):
    contract, context = make_contract_and_context()
    features = tuple(
        f.model_copy(
            update={
                "constraints": FeatureConstraints(
                    minimum=0.0,
                    maximum=10.0,
                    minimum_inclusive=False,
                    maximum_inclusive=False,
                )
            }
        )
        if f.name == "speed_per_kg"
        else f
        for f in contract.features
    )
    contract = DatasetContract.model_validate({**contract.model_dump(), "features": features})
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    spec = DerivedFeatureSpec(feature_name="speed_per_kg", transform=ConstantValue(value=value))
    res = DeterministicFeatureEngineer(storage, derived_specs=(spec,)).engineer(
        context=context, artifact=cleaned, contract=contract
    )
    assert res.status == expected_status


def test_derived_feature_allowed_categorical_values_and_no_exec_after_structural_block(
    tmp_path: Path,
):
    contract, context = make_contract_and_context()
    cat_feature = FeatureContract(
        name="speed_band",
        display_name="Speed Band",
        description="Categorical band.",
        data_type=DataType.CATEGORY,
        role=FeatureRole.PREDICTOR,
        availability=FeatureAvailability.PRE_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=True,
        nullable=False,
        missing_policy=MissingValuePolicy.REJECT,
        constraints=FeatureConstraints(allowed_values=("LOW", "HIGH")),
        derived_from=("machine_speed",),
    )
    contract = DatasetContract.model_validate(
        {
            **contract.model_dump(),
            "features": (*contract.features, cat_feature),
        }
    )
    storage = LocalDatasetArtifactStorage(tmp_path)
    mapped = map_csv(
        storage,
        context,
        contract,
        b"batch_id,planned_qty,machine_speed,waste_qty\nB1,100,300,2\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=context, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    # Invalid category value is blocked
    res_bad = DeterministicFeatureEngineer(
        storage,
        derived_specs=(
            DerivedFeatureSpec(
                feature_name="speed_per_kg",
                transform=BinaryExpr(
                    op=BinaryOperator.DIV,
                    left=ColumnRef(name="machine_speed"),
                    right=ColumnRef(name="planned_qty"),
                ),
            ),
            DerivedFeatureSpec(feature_name="speed_band", transform=ConstantValue(value="INVALID")),
        ),
    ).engineer(context=context, artifact=cleaned, contract=contract)
    assert res_bad.status == ExecutionStatus.BLOCKED
    assert res_bad.report is not None
    assert any(i.code == "CONSTRAINT_ALLOWED_VALUES_VIOLATION" for i in res_bad.report.issues)

    # Structural block (missing required spec for speed_band) prevents ANY callback from running
    calls = 0

    def _spy(frame: pd.DataFrame) -> pd.Series:
        nonlocal calls
        calls += 1
        return frame["machine_speed"] / frame["planned_qty"]

    res_struct = DeterministicFeatureEngineer(
        storage,
        derived_specs=(DerivedFeatureSpec("speed_per_kg", _spy),),
        allow_unverified_callbacks=True,
    ).engineer(context=context, artifact=cleaned, contract=contract)
    assert res_struct.status == ExecutionStatus.BLOCKED
    assert calls == 0


def test_prediction_time_unavailable_derived_feature_blocked_when_supplied(tmp_path: Path):
    contract, train_ctx = make_contract_and_context()
    post_derived = FeatureContract(
        name="post_prod_derived",
        display_name="Post Prod Derived",
        description="Unavailable at prediction.",
        data_type=DataType.FLOAT,
        role=FeatureRole.KPI,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.DERIVED,
        required=False,
        nullable=True,
        missing_policy=MissingValuePolicy.ALLOW,
        derived_from=("planned_qty",),
    )
    contract = DatasetContract.model_validate(
        {**contract.model_dump(), "features": (*contract.features, post_derived)}
    )
    from uuid import uuid4

    pred_ctx = train_ctx.model_copy(
        update={"purpose": DatasetPurpose.PREDICTION, "model_version_id": uuid4()}
    )
    storage = LocalDatasetArtifactStorage(tmp_path)

    mapped = map_csv(
        storage,
        pred_ctx,
        contract,
        b"batch_id,planned_qty,machine_speed\nB1,100,300\n",
    )
    cleaned = (
        ContractDataCleaner(storage)
        .clean(context=pred_ctx, artifact=mapped, contract=contract)
        .output_artifact
    )
    assert cleaned is not None

    payload = b"batch_id,planned_qty,machine_speed,post_prod_derived\nB1,100,300,99.0\n"
    ref = ArtifactReference(
        factory_id=pred_ctx.factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    storage.put(factory_id=pred_ctx.factory_id, reference=ref, stream=BytesIO(payload))
    meta = cleaned.metadata.model_copy(
        update={
            "columns": (
                *cleaned.metadata.columns,
                DatasetColumnSchema(name="post_prod_derived", physical_dtype="Float64"),
            )
        }
    )
    artifact = DatasetArtifact(metadata=meta, reference=ref)

    specs = (
        DerivedFeatureSpec(
            feature_name="speed_per_kg",
            transform=BinaryExpr(
                op=BinaryOperator.DIV,
                left=ColumnRef(name="machine_speed"),
                right=ColumnRef(name="planned_qty"),
            ),
        ),
        DerivedFeatureSpec(
            feature_name="post_prod_derived",
            transform=ConstantValue(value=1.0),
        ),
    )
    res = DeterministicFeatureEngineer(storage, derived_specs=specs).engineer(
        context=pred_ctx, artifact=artifact, contract=contract
    )
    assert res.status == ExecutionStatus.BLOCKED
    assert res.output_artifact is None
    assert res.report is not None
    assert any(i.code == "UNAVAILABLE_FEATURE_IN_PREDICTION" for i in res.report.issues)
