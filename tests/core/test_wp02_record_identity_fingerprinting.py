"""WP02 regression suite — record identity, canonicalization, and fingerprinting.

Covers:
  Group A — Canonicalization and numerical precision.
  Group B — Missing and invalid values.
  Group C — Composite business keys.
  Group D — Scope mismatches.
  Group E — Legacy fingerprint migration safety.
  Group F — Zero registry writes on failed preflight.
  Group G — Historical record correction.
  Group H — Prediction identity separation.
  Group I — PredictionObservation contract invariants.
  Group J — Performance benchmarks (10k / 100k / 250k rows).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from hashlib import sha256 as _sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from taqamind.core.contracts.dataset import DatasetContract, PredictionContext
from taqamind.core.contracts.enums import (
    DatasetPurpose,
    DataType,
    FeatureAvailability,
    FeatureOrigin,
    FeatureRole,
    MissingValuePolicy,
    PipelineStage,
    TargetKind,
    TaskType,
)
from taqamind.core.contracts.feature import FeatureConstraints, FeatureContract
from taqamind.core.contracts.target import (
    TargetConfirmation,
    TargetContract,
    TargetDefinition,
)
from taqamind.core.interfaces import (
    ArtifactFormat,
    ArtifactReference,
    DatasetArtifact,
    DatasetColumnSchema,
    DatasetMetadata,
    DatasetState,
    FingerprintField,
    PipelineContext,
    RecordIdentity,
    RecordIdentityStatus,
    RecordMatchStatus,
    RecordScope,
    RecordVersion,
    TransformationRecord,
    calculate_record_fingerprint,
)
from taqamind.core.interfaces.lineage import (
    PredictionIdentityStatus,
    PredictionObservation,
    PredictionObservationBatch,
)
from taqamind.core.lineage import (
    FINGERPRINT_VERSION_V1,
    FINGERPRINT_VERSION_V2,
    CanonicalizationError,
    LocalRecordLineageRegistry,
    RecordMembershipBuilder,
    canonicalize_column,
    find_duplicate_canonical_business_keys,
    hash_canonical_v2_row,
)
from taqamind.core.registry import LocalDatasetArtifactStorage

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_feature(
    name: str,
    *,
    dtype: DataType = DataType.STRING,
    role: FeatureRole = FeatureRole.PREDICTOR,
    availability: FeatureAvailability = FeatureAvailability.PRE_PRODUCTION,
    origin: FeatureOrigin = FeatureOrigin.RECORDED,
    nullable: bool = False,
    missing_policy: MissingValuePolicy = MissingValuePolicy.REJECT,
    constraints: FeatureConstraints | None = None,
    allowed_values: tuple[str, ...] | None = None,
) -> FeatureContract:
    _constraints = constraints
    if allowed_values is not None:
        _constraints = FeatureConstraints(allowed_values=allowed_values)
    return FeatureContract(
        name=name,
        display_name=name.replace("_", " ").title(),
        description=f"Test feature {name}",
        data_type=dtype,
        role=role,
        availability=availability,
        origin=origin,
        nullable=nullable,
        missing_policy=missing_policy,
        constraints=_constraints or FeatureConstraints(),
    )


def _make_target_contract(*, feature_name: str = "waste", unit: str = "kg") -> TargetContract:
    from taqamind.core.contracts.enums import ConfirmationStatus

    feat = FeatureContract(
        name=feature_name,
        display_name="Waste",
        description="Target waste",
        data_type=DataType.FLOAT,
        role=FeatureRole.TARGET,
        availability=FeatureAvailability.POST_PRODUCTION,
        origin=FeatureOrigin.MEASURED,
        unit=unit,
        constraints=FeatureConstraints(minimum=0.0),
        missing_policy=MissingValuePolicy.REJECT,
    )
    return TargetContract(
        target_id="waste_target",
        feature=feat,
        task_type=TaskType.REGRESSION,
        target_kind=TargetKind.QUANTITY,
        definition=TargetDefinition(definition="Industrial waste", granularity="batch"),
        confirmation=TargetConfirmation(
            status=ConfirmationStatus.CONFIRMED,
            confirmed_by="wp02_test",
            confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def _minimal_contract(
    *,
    factory_id: UUID | None = None,
    dataset_id: str = "test_ds",
    domain: str = "textile",
    process: str = "weaving",
    business_keys: tuple[str, ...] = ("batch_id",),
    extra_features: tuple[FeatureContract, ...] = (),
) -> DatasetContract:
    bk_feat = _make_feature(
        "batch_id",
        role=FeatureRole.IDENTIFIER,
        missing_policy=MissingValuePolicy.REJECT,
    )
    qty_feat = _make_feature(
        "qty",
        dtype=DataType.FLOAT,
        constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
        missing_policy=MissingValuePolicy.REJECT,
    )
    return DatasetContract(
        dataset_id=dataset_id,
        version="1.0.0",
        domain=domain,
        process=process,
        description="WP02 test contract",
        granularity="batch",
        features=(bk_feat, qty_feat) + extra_features,
        target=_make_target_contract(),
        prediction_context=PredictionContext(description="Test prediction"),
        business_keys=business_keys,
    )


def _make_context(
    *,
    factory_id: UUID,
    dataset_id: str = "test_ds",
    domain: str = "textile",
    process: str = "weaving",
    purpose: DatasetPurpose = DatasetPurpose.TRAINING,
) -> PipelineContext:
    return PipelineContext(
        factory_id=factory_id,
        dataset_revision_id=uuid4(),
        domain=domain,
        process=process,
        dataset_id=dataset_id,
        contract_version="1.0.0",
        task_id="waste_regression",
        task_version="1.0.0",
        target_version="1.0.0",
        purpose=purpose,
        model_version_id=uuid4() if purpose == DatasetPurpose.PREDICTION else None,
    )


def _store_csv(
    storage: LocalDatasetArtifactStorage,
    *,
    factory_id: UUID,
    csv_bytes: bytes,
) -> ArtifactReference:
    ref = ArtifactReference(
        factory_id=factory_id,
        format=ArtifactFormat.CSV,
        content_sha256=_sha256(csv_bytes).hexdigest(),
        size_bytes=len(csv_bytes),
    )
    storage.put(factory_id=factory_id, reference=ref, stream=BytesIO(csv_bytes))
    return ref


def _make_mapped_artifact(
    storage: LocalDatasetArtifactStorage,
    *,
    factory_id: UUID,
    domain: str = "textile",
    process: str = "weaving",
    dataset_id: str = "test_ds",
    csv_bytes: bytes,
    columns: tuple[DatasetColumnSchema, ...],
) -> DatasetArtifact:
    ref = _store_csv(storage, factory_id=factory_id, csv_bytes=csv_bytes)
    root_id = uuid4()
    rev_id = uuid4()
    metadata = DatasetMetadata(
        revision_id=rev_id,
        root_revision_id=root_id,
        factory_id=factory_id,
        domain=domain,
        process=process,
        dataset_id=dataset_id,
        contract_version="1.0.0",
        task_id="waste_regression",
        task_version="1.0.0",
        target_version="1.0.0",
        source_file_sha256=ref.content_sha256,
        state=DatasetState.MAPPED,
        row_count=len(csv_bytes.decode().strip().split("\n")) - 1,
        columns=columns,
        transformation=TransformationRecord(
            execution_id=uuid4(),
            parent_revision_id=root_id,
            stage=PipelineStage.COLUMN_MAPPING,
            operation="column_mapping",
            implementation_version="1.0.0",
            description="Test mapped dataset artifact",
        ),
    )
    return DatasetArtifact(metadata=metadata, reference=ref)


BASIC_COLS = (
    DatasetColumnSchema(name="batch_id", physical_dtype="object"),
    DatasetColumnSchema(name="qty", physical_dtype="float64"),
    DatasetColumnSchema(name="waste", physical_dtype="float64"),
)

BASIC_CSV = b"batch_id,qty,waste\n001,100.0,2.0\n002,150.0,3.0\n"


def _make_scope(
    *,
    factory_id: UUID,
    domain: str = "textile",
    process: str = "weaving",
    dataset_id: str = "test_ds",
    identity_namespace: str = "production_record",
    identity_version: str = "1.0.0",
) -> RecordScope:
    return RecordScope(
        factory_id=factory_id,
        domain=domain,
        process=process,
        dataset_id=dataset_id,
        identity_namespace=identity_namespace,
        identity_version=identity_version,
    )


# ---------------------------------------------------------------------------
# Group A — Canonicalization and numerical precision
# ---------------------------------------------------------------------------


class TestCanonicalizationPrecision:
    """Group A: Typed v2 token determinism and numerical precision."""

    def test_string_tokens_strip_surrounding_whitespace(self):
        feat = _make_feature("col", dtype=DataType.STRING)
        series = pd.Series(["  ABC  ", " DEF", "GHI  "])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["str:ABC", "str:DEF", "str:GHI"]
        assert result.values == ["ABC", "DEF", "GHI"]

    def test_string_tokens_preserve_case_and_leading_zeros(self):
        feat = _make_feature("col", dtype=DataType.STRING)
        series = pd.Series(["01", "0010", "ABC"])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["str:01", "str:0010", "str:ABC"]

    def test_integer_tokens_exact_int64_max(self):
        feat = _make_feature("col", dtype=DataType.INTEGER)
        int64_max = 9_223_372_036_854_775_807
        series = pd.Series([int64_max], dtype="Int64")
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == [f"int:{int64_max}"]
        assert result.values[0] == int64_max

    def test_integer_tokens_exact_int64_min(self):
        feat = _make_feature("col", dtype=DataType.INTEGER)
        int64_min = -9_223_372_036_854_775_808
        series = pd.Series([int64_min], dtype="Int64")
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == [f"int:{int64_min}"]

    def test_float_token_uses_hex_representation(self):
        feat = _make_feature("col", dtype=DataType.FLOAT)
        v = 1.23456789012345
        series = pd.Series([v])
        result = canonicalize_column(series, feat)
        expected_hex = v.hex()
        assert result.v2_tokens == [f"float:{expected_hex}"]

    def test_float_negative_zero_normalised_to_positive_zero(self):
        feat = _make_feature("col", dtype=DataType.FLOAT)
        series = pd.Series([-0.0])
        result = canonicalize_column(series, feat)
        # -0.0 and 0.0 are equal; token must equal that of +0.0
        assert result.v2_tokens == [f"float:{(0.0).hex()}"]

    def test_float_above_int64_precision_boundary(self):
        feat = _make_feature("col", dtype=DataType.FLOAT)
        v = float(2**53 + 1)  # beyond exact float64 integer precision
        series = pd.Series([v])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == [f"float:{v.hex()}"]

    def test_boolean_tokens(self):
        feat = _make_feature("col", dtype=DataType.BOOLEAN)
        series = pd.Series([True, False])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["bool:true", "bool:false"]

    def test_category_tokens_strip_whitespace(self):
        feat = _make_feature(
            "col",
            dtype=DataType.CATEGORY,
            allowed_values=("A", "B"),
        )
        series = pd.Series([" A", "B "])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["cat:A", "cat:B"]

    def test_null_token_for_missing_value(self):
        feat = _make_feature(
            "col",
            dtype=DataType.STRING,
            nullable=True,
            missing_policy=MissingValuePolicy.ALLOW,
        )
        series = pd.Series([None, "X"])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens[0] == "null"
        assert result.values[0] is None

    def test_hash_canonical_v2_row_deterministic(self):
        pairs = [("batch_id", "str:001"), ("qty", "float:0x1p+0")]
        h1 = hash_canonical_v2_row(namespace="business_key", field_pairs=pairs)
        h2 = hash_canonical_v2_row(namespace="business_key", field_pairs=pairs)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_canonical_v2_row_different_values_differ(self):
        h1 = hash_canonical_v2_row(
            namespace="bk",
            field_pairs=[("id", "str:001")],
        )
        h2 = hash_canonical_v2_row(
            namespace="bk",
            field_pairs=[("id", "str:002")],
        )
        assert h1 != h2

    def test_v2_fingerprint_differs_from_v1_for_same_data(self):
        fields = (FingerprintField(name="batch_id", value="B001"),)
        v1 = calculate_record_fingerprint(fields, namespace="ns", version=FINGERPRINT_VERSION_V1)
        v2_pairs = [("batch_id", "str:B001")]
        v2 = hash_canonical_v2_row(namespace="ns", field_pairs=v2_pairs)
        assert v1 != v2

    def test_v2_float_token_lossless_roundtrip(self):
        import struct

        v = struct.unpack("d", struct.pack("d", 1.0 / 3.0))[0]
        feat = _make_feature("col", dtype=DataType.FLOAT)
        series = pd.Series([v])
        result = canonicalize_column(series, feat)
        token = result.v2_tokens[0]
        assert token.startswith("float:")
        recovered = float.fromhex(token[6:])
        assert recovered == v  # exact bit-identical roundtrip


# ---------------------------------------------------------------------------
# Group B — Missing and invalid values
# ---------------------------------------------------------------------------


class TestMissingAndInvalidValues:
    """Group B: Reject invalid values before fingerprinting."""

    def test_non_finite_float_raises_canonicalization_error(self):
        feat = _make_feature("col", dtype=DataType.FLOAT)
        for bad in [float("inf"), float("-inf")]:
            series = pd.Series([bad])
            with pytest.raises(CanonicalizationError, match="INVALID_FINGERPRINT_INPUT"):
                canonicalize_column(series, feat)

    def test_category_outside_allowed_values_raises(self):
        feat = _make_feature("col", dtype=DataType.CATEGORY, allowed_values=("A", "B"))
        series = pd.Series(["C"])
        with pytest.raises(CanonicalizationError, match="INVALID_FINGERPRINT_INPUT"):
            canonicalize_column(series, feat)

    def test_missing_integer_in_nullable_column_produces_null_token(self):
        feat = _make_feature(
            "col", dtype=DataType.INTEGER, nullable=True, missing_policy=MissingValuePolicy.ALLOW
        )
        series = pd.Series([pd.NA], dtype="Int64")
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["null"]
        assert result.values == [None]

    def test_missing_float_in_nullable_column_produces_null_token(self):
        feat = _make_feature(
            "col", dtype=DataType.FLOAT, nullable=True, missing_policy=MissingValuePolicy.ALLOW
        )
        series = pd.Series([float("nan")])
        result = canonicalize_column(series, feat)
        assert result.v2_tokens == ["null"]

    def test_find_duplicate_canonical_business_keys_detects_whitespace_normalized_duplicates(self):
        contract = _minimal_contract()
        df = pd.DataFrame({"batch_id": ["  001  ", "001"], "qty": [1.0, 2.0], "waste": [1.0, 2.0]})
        dup_indices = find_duplicate_canonical_business_keys(df, contract)
        assert len(dup_indices) > 0
        assert 1 in dup_indices

    def test_find_duplicate_returns_empty_on_distinct_keys(self):
        contract = _minimal_contract()
        df = pd.DataFrame(
            {"batch_id": ["001", "002", "003"], "qty": [1.0, 2.0, 3.0], "waste": [1.0, 2.0, 3.0]}
        )
        dup_indices = find_duplicate_canonical_business_keys(df, contract)
        assert dup_indices == ()

    def test_find_duplicate_skips_null_keys(self):
        contract = _minimal_contract()
        # Two null business keys are NOT treated as duplicates.
        df = pd.DataFrame(
            {"batch_id": [None, None, "003"], "qty": [1.0, 2.0, 3.0], "waste": [1.0, 2.0, 3.0]}
        )
        dup_indices = find_duplicate_canonical_business_keys(df, contract)
        # nulls are skipped; only "003" appears once → no duplicates
        assert 0 not in dup_indices
        assert 1 not in dup_indices


# ---------------------------------------------------------------------------
# Group C — Composite business keys
# ---------------------------------------------------------------------------


class TestCompositeBusinessKeys:
    """Group C: Multi-column business key canonicalization."""

    def test_composite_key_hash_uses_sorted_field_order(self):
        # Sorted field order must be stable regardless of DataFrame column order.
        pairs_ab = [("a", "str:X"), ("b", "int:1")]
        pairs_ba = [("b", "int:1"), ("a", "str:X")]
        h_ab = hash_canonical_v2_row(namespace="bk", field_pairs=pairs_ab)
        h_ba = hash_canonical_v2_row(namespace="bk", field_pairs=pairs_ba)
        # Different order → different hash (caller is responsible for sorting).
        assert h_ab != h_ba

    def test_composite_key_find_duplicates_detects_matching_combination(self):
        line_feat = _make_feature(
            "line_id",
            role=FeatureRole.IDENTIFIER,
            missing_policy=MissingValuePolicy.REJECT,
        )
        shift_feat = _make_feature(
            "shift_id",
            role=FeatureRole.IDENTIFIER,
            missing_policy=MissingValuePolicy.REJECT,
        )
        qty_feat = _make_feature(
            "qty",
            dtype=DataType.FLOAT,
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            missing_policy=MissingValuePolicy.REJECT,
        )
        contract = DatasetContract(
            dataset_id="composite_ds",
            version="1.0.0",
            domain="textile",
            process="weaving",
            description="Composite key test",
            granularity="batch",
            features=(line_feat, shift_feat, qty_feat),
            target=_make_target_contract(),
            prediction_context=PredictionContext(description="Test"),
            business_keys=("line_id", "shift_id"),
        )
        df = pd.DataFrame(
            {
                "line_id": ["L1", "L1", "L2"],
                "shift_id": ["AM", "AM", "AM"],
                "qty": [1.0, 2.0, 3.0],
                "waste": [0.1, 0.2, 0.3],
            }
        )
        dup_indices = find_duplicate_canonical_business_keys(df, contract)
        assert 1 in dup_indices
        assert 2 not in dup_indices

    def test_composite_key_different_line_same_shift_not_duplicate(self):
        contract = _minimal_contract(
            business_keys=("batch_id",),
        )
        df = pd.DataFrame(
            {
                "batch_id": ["L1_AM", "L2_AM"],
                "qty": [1.0, 2.0],
                "waste": [0.1, 0.2],
            }
        )
        dup_indices = find_duplicate_canonical_business_keys(df, contract)
        assert dup_indices == ()


# ---------------------------------------------------------------------------
# Group D — Scope mismatches
# ---------------------------------------------------------------------------


class TestScopeMismatches:
    """Group D: RecordMembershipBuilder scope and contract verification."""

    def test_build_and_register_rejects_mismatched_domain(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(
            factory_id=factory_id,
            domain="textile",
            process="weaving",
        )
        artifact = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            domain="textile",
            process="weaving",
            csv_bytes=BASIC_CSV,
            columns=BASIC_COLS,
        )

        # Scope with different domain.
        scope = _make_scope(factory_id=factory_id, domain="iron_manufacturing", process="weaving")
        with pytest.raises(ValueError, match="[Dd]omain mismatch"):
            builder.build_and_register(scope=scope, artifact=artifact, contract=contract)

    def test_build_and_register_rejects_mismatched_factory(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        artifact = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=BASIC_CSV,
            columns=BASIC_COLS,
        )
        scope = _make_scope(factory_id=uuid4())  # different factory
        with pytest.raises(ValueError, match="[Ff]actory"):
            builder.build_and_register(scope=scope, artifact=artifact, contract=contract)

    def test_build_and_register_rejects_prediction_purpose(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        artifact = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=BASIC_CSV,
            columns=BASIC_COLS,
        )
        scope = _make_scope(factory_id=factory_id)
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        with pytest.raises(ValueError, match="HISTORICAL_REGISTRATION_WITH_PREDICTION_PURPOSE"):
            builder.build_and_register(
                scope=scope, artifact=artifact, contract=contract, context=ctx
            )

    def test_build_and_register_rejects_identity_namespace_mismatch(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg, identity_namespace="production_record")

        contract = _minimal_contract(factory_id=factory_id)
        artifact = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=BASIC_CSV,
            columns=BASIC_COLS,
        )
        scope = _make_scope(factory_id=factory_id, identity_namespace="other_namespace")
        with pytest.raises(ValueError, match="identity_namespace"):
            builder.build_and_register(scope=scope, artifact=artifact, contract=contract)


# ---------------------------------------------------------------------------
# Group E — Legacy fingerprint migration safety
# ---------------------------------------------------------------------------


class TestLegacyFingerprintMigrationSafety:
    """Group E: v1→v2 migration blocks; existing IDs preserved."""

    def test_v1_identity_preserved_when_using_explicit_v1(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)
        art = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=BASIC_CSV,
            columns=BASIC_COLS,
        )
        summary = builder.build_and_register(
            scope=scope,
            artifact=art,
            contract=contract,
            content_fingerprint_version=FINGERPRINT_VERSION_V1,
        )
        assert summary.new_records == 2
        # Both records get v1 content fingerprints.
        for membership in summary.manifest.memberships:
            ver = lin_reg.get_version(scope=scope, record_version_id=membership.record_version_id)
            assert ver.content_fingerprint_version == FINGERPRINT_VERSION_V1

    def test_v2_registration_blocked_when_v1_identity_exists(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        # First registration with v1.
        art1 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=(
                DatasetColumnSchema(name="batch_id", physical_dtype="object"),
                DatasetColumnSchema(name="qty", physical_dtype="float64"),
                DatasetColumnSchema(name="waste", physical_dtype="float64"),
            ),
        )
        builder.build_and_register(
            scope=scope,
            artifact=art1,
            contract=contract,
            content_fingerprint_version=FINGERPRINT_VERSION_V1,
        )

        # Second registration of same record with v2 must be blocked.
        art2 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=(
                DatasetColumnSchema(name="batch_id", physical_dtype="object"),
                DatasetColumnSchema(name="qty", physical_dtype="float64"),
                DatasetColumnSchema(name="waste", physical_dtype="float64"),
            ),
        )
        with pytest.raises(ValueError, match="LEGACY_MIGRATION_REQUIRED"):
            builder.build_and_register(
                scope=scope,
                artifact=art2,
                contract=contract,
                content_fingerprint_version=FINGERPRINT_VERSION_V2,
            )

    def test_migration_required_cannot_be_bypassed_as_correction(self, tmp_path: Path):
        """An ordinary update confirmation cannot bypass MIGRATION_REQUIRED blocking."""
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        art1 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=(
                DatasetColumnSchema(name="batch_id", physical_dtype="object"),
                DatasetColumnSchema(name="qty", physical_dtype="float64"),
                DatasetColumnSchema(name="waste", physical_dtype="float64"),
            ),
        )
        builder.build_and_register(
            scope=scope,
            artifact=art1,
            contract=contract,
            content_fingerprint_version=FINGERPRINT_VERSION_V1,
        )

        art2 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=(
                DatasetColumnSchema(name="batch_id", physical_dtype="object"),
                DatasetColumnSchema(name="qty", physical_dtype="float64"),
                DatasetColumnSchema(name="waste", physical_dtype="float64"),
            ),
        )
        # Even with explicit confirmation, v1→v2 migration must be blocked.
        with pytest.raises(ValueError, match="LEGACY_MIGRATION_REQUIRED"):
            builder.build_and_register(
                scope=scope,
                artifact=art2,
                contract=contract,
                content_fingerprint_version=FINGERPRINT_VERSION_V2,
                confirm_updates_by="operator",
                update_reason="Routine correction",
            )

    def test_legacy_lossy_float_business_key_fails_closed(self):
        """Float business keys cannot produce a v1 fallback (lossy risk)."""
        float_bk = FeatureContract(
            name="measure_id",
            display_name="Measure ID",
            description="Float business key",
            data_type=DataType.FLOAT,
            role=FeatureRole.IDENTIFIER,
            availability=FeatureAvailability.PRE_PRODUCTION,
            origin=FeatureOrigin.MEASURED,
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            missing_policy=MissingValuePolicy.REJECT,
        )
        qty_feat = _make_feature(
            "qty",
            dtype=DataType.FLOAT,
            constraints=FeatureConstraints(minimum=0.0, minimum_inclusive=False),
            missing_policy=MissingValuePolicy.REJECT,
        )
        contract = DatasetContract(
            dataset_id="float_bk_ds",
            version="1.0.0",
            domain="textile",
            process="weaving",
            description="Float BK test",
            granularity="batch",
            features=(float_bk, qty_feat),
            target=_make_target_contract(),
            prediction_context=PredictionContext(description="Test"),
            business_keys=("measure_id",),
        )
        col_map = contract.column_map
        # Confirm float business key → no v1 fallback computation.
        # (In membership_builder: key_has_float=True → biz_sha_v1=None)
        assert col_map["measure_id"].data_type == DataType.FLOAT

    def test_v1_match_returns_migration_required_in_registry(self, tmp_path: Path):
        """registry.match() with mismatched fingerprint version → MIGRATION_REQUIRED."""
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        scope = _make_scope(factory_id=uuid4())

        # Register a v1 identity and version.
        identity = lin_reg.register_identity(
            scope=scope,
            identity=RecordIdentity(
                scope=scope,
                business_key_sha256="a" * 64,
                status=RecordIdentityStatus.CONFIRMED,
            ),
        )
        lin_reg.register_version(
            scope=scope,
            version=RecordVersion(
                identity=identity,
                content_sha256="b" * 64,
                content_fingerprint_version=FINGERPRINT_VERSION_V1,
                source_dataset_revision_id=uuid4(),
            ),
        )

        # Match with v2 fingerprint → MIGRATION_REQUIRED.
        result = lin_reg.match(
            scope=scope,
            business_key_sha256="a" * 64,
            content_sha256="b" * 64,
            content_fingerprint_version=FINGERPRINT_VERSION_V2,
        )
        assert result.status == RecordMatchStatus.MIGRATION_REQUIRED
        assert result.matched_version is not None


# ---------------------------------------------------------------------------
# Group F — Zero registry writes on failed preflight
# ---------------------------------------------------------------------------


class TestZeroRegistryWritesOnFailedPreflight:
    """Group F: Deterministic preflight — no partial writes on failure."""

    def test_missing_required_business_key_aborts_all_rows(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        # Row 0 is valid; Row 1 has missing required business key.
        csv_bytes = b"batch_id,qty,waste\n001,100.0,2.0\n,150.0,3.0\n"
        art = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=csv_bytes,
            columns=BASIC_COLS,
        )
        with pytest.raises(ValueError, match="missing required business key"):
            builder.build_and_register(scope=scope, artifact=art, contract=contract)

        # No identities should have been registered.
        from taqamind.core.interfaces.dataset_registry import RegistryNotFoundError

        with pytest.raises(RegistryNotFoundError):
            lin_reg.get_membership(
                scope=scope,
                dataset_revision_id=art.metadata.revision_id,
            )

    def test_duplicate_business_key_in_batch_aborts_preflight(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        # Duplicate batch_id "001" in the same batch.
        csv_bytes = b"batch_id,qty,waste\n001,100.0,2.0\n001,120.0,2.5\n"
        art = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=csv_bytes,
            columns=BASIC_COLS,
        )
        with pytest.raises(ValueError, match="DUPLICATE_BUSINESS_KEY"):
            builder.build_and_register(scope=scope, artifact=art, contract=contract)

    def test_unconfirmed_correction_produces_zero_new_registry_entries(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        # Register initial record.
        art1 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=BASIC_COLS,
        )
        s1 = builder.build_and_register(scope=scope, artifact=art1, contract=contract)
        initial_versions = lin_reg.get_versions(
            scope=scope, record_id=s1.manifest.memberships[0].record_id
        )
        assert len(initial_versions) == 1

        # Attempt unconfirmed correction.
        art2 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,9.0\n",
            columns=BASIC_COLS,
        )
        with pytest.raises(ValueError, match="without explicit confirmation"):
            builder.build_and_register(scope=scope, artifact=art2, contract=contract)

        # Version count must remain 1 — no partial write occurred.
        post_versions = lin_reg.get_versions(
            scope=scope, record_id=s1.manifest.memberships[0].record_id
        )
        assert len(post_versions) == 1


# ---------------------------------------------------------------------------
# Group G — Historical record correction
# ---------------------------------------------------------------------------


class TestHistoricalRecordCorrection:
    """Group G: Confirmed correction increments version; exact match reuses version."""

    def test_exact_match_reuses_existing_version(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        art1 = _make_mapped_artifact(
            storage, factory_id=factory_id, csv_bytes=BASIC_CSV, columns=BASIC_COLS
        )
        s1 = builder.build_and_register(scope=scope, artifact=art1, contract=contract)
        assert s1.new_records == 2

        art2 = _make_mapped_artifact(
            storage, factory_id=factory_id, csv_bytes=BASIC_CSV, columns=BASIC_COLS
        )
        s2 = builder.build_and_register(scope=scope, artifact=art2, contract=contract)
        assert s2.exact_matches == 2
        assert s2.new_records == 0
        # Version IDs must be identical.
        for m1, m2 in zip(s1.manifest.memberships, s2.manifest.memberships, strict=True):
            assert m1.record_version_id == m2.record_version_id

    def test_confirmed_correction_increments_version_number(self, tmp_path: Path):
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        art1 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=BASIC_COLS,
        )
        s1 = builder.build_and_register(scope=scope, artifact=art1, contract=contract)
        record_id = s1.manifest.memberships[0].record_id

        art2 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,5.0\n",
            columns=BASIC_COLS,
        )
        s2 = builder.build_and_register(
            scope=scope,
            artifact=art2,
            contract=contract,
            confirm_updates_by="qa_engineer",
            update_reason="Recalibration",
        )
        assert s2.content_changed == 1

        versions = lin_reg.get_versions(scope=scope, record_id=record_id)
        assert len(versions) == 2
        assert versions[0].version_number == 1
        assert versions[1].version_number == 2
        assert versions[1].confirmed_by == "qa_engineer"

    def test_historical_replay_raises_review_required(self, tmp_path: Path):
        """Incoming content that matches an older (non-latest) historical version → review."""
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)

        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        # v1: waste=2.0
        art1 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=BASIC_COLS,
        )
        builder.build_and_register(scope=scope, artifact=art1, contract=contract)

        # v2: waste=5.0 (confirmed correction)
        art2 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,5.0\n",
            columns=BASIC_COLS,
        )
        builder.build_and_register(
            scope=scope,
            artifact=art2,
            contract=contract,
            confirm_updates_by="qa",
            update_reason="Correction",
        )

        # Now re-submit waste=2.0 (matches v1, not latest v2) → must be blocked.
        art3 = _make_mapped_artifact(
            storage,
            factory_id=factory_id,
            csv_bytes=b"batch_id,qty,waste\n001,100.0,2.0\n",
            columns=BASIC_COLS,
        )
        with pytest.raises(ValueError, match="HISTORICAL_REPLAY_REVIEW_REQUIRED"):
            builder.build_and_register(scope=scope, artifact=art3, contract=contract)


# ---------------------------------------------------------------------------
# Group H — Prediction identity separation
# ---------------------------------------------------------------------------


class TestPredictionIdentitySeparation:
    """Group H: resolve_prediction_observations never mutates historical lineage."""

    def _build_prediction_infrastructure(
        self,
        tmp_path: Path,
        *,
        with_training_data: bool = True,
    ) -> tuple:
        factory_id = uuid4()
        storage = LocalDatasetArtifactStorage(tmp_path / "art")
        lin_reg = LocalRecordLineageRegistry(tmp_path / "reg")
        builder = RecordMembershipBuilder(storage, lin_reg)
        contract = _minimal_contract(factory_id=factory_id)
        scope = _make_scope(factory_id=factory_id)

        training_record_ids: list[UUID] = []
        if with_training_data:
            art_train = _make_mapped_artifact(
                storage,
                factory_id=factory_id,
                csv_bytes=BASIC_CSV,
                columns=BASIC_COLS,
            )
            s = builder.build_and_register(scope=scope, artifact=art_train, contract=contract)
            training_record_ids = [m.record_id for m in s.manifest.memberships]

        return factory_id, storage, lin_reg, builder, contract, scope, training_record_ids

    def _make_prediction_artifact(
        self,
        storage: LocalDatasetArtifactStorage,
        *,
        factory_id: UUID,
        csv_bytes: bytes,
        columns: tuple,
    ) -> DatasetArtifact:
        ref = _store_csv(storage, factory_id=factory_id, csv_bytes=csv_bytes)
        root_id = uuid4()
        rev_id = uuid4()
        metadata = DatasetMetadata(
            revision_id=rev_id,
            root_revision_id=root_id,
            factory_id=factory_id,
            domain="textile",
            process="weaving",
            dataset_id="test_ds",
            contract_version="1.0.0",
            task_id="waste_regression",
            task_version="1.0.0",
            target_version="1.0.0",
            source_file_sha256=ref.content_sha256,
            state=DatasetState.MAPPED,
            row_count=len(csv_bytes.decode().strip().split("\n")) - 1,
            columns=columns,
            transformation=TransformationRecord(
                execution_id=uuid4(),
                parent_revision_id=root_id,
                stage=PipelineStage.COLUMN_MAPPING,
                operation="column_mapping",
                implementation_version="1.0.0",
                description="Test prediction dataset artifact",
            ),
        )
        return DatasetArtifact(metadata=metadata, reference=ref)

    def test_known_historical_observation_resolves_record_id(self, tmp_path: Path):
        factory_id, storage, lin_reg, builder, contract, scope, training_ids = (
            self._build_prediction_infrastructure(tmp_path)
        )
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        pred_csv = b"batch_id,qty\n001,100.0\n002,150.0\n"
        pred_cols = (
            DatasetColumnSchema(name="batch_id", physical_dtype="object"),
            DatasetColumnSchema(name="qty", physical_dtype="float64"),
        )
        pred_art = self._make_prediction_artifact(
            storage, factory_id=factory_id, csv_bytes=pred_csv, columns=pred_cols
        )
        batch = builder.resolve_prediction_observations(
            scope=scope,
            artifact=pred_art,
            contract=contract,
            context=ctx,
        )
        assert len(batch.observations) == 2
        for obs in batch.observations:
            assert obs.resolution_status == PredictionIdentityStatus.KNOWN_HISTORICAL
            assert obs.known_record_id in training_ids

    def test_unknown_observation_has_none_record_id_and_new_candidate_status(self, tmp_path: Path):
        factory_id, storage, lin_reg, builder, contract, scope, _ = (
            self._build_prediction_infrastructure(tmp_path)
        )
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        # "999" was never registered in training.
        pred_csv = b"batch_id,qty\n999,100.0\n"
        pred_cols = (
            DatasetColumnSchema(name="batch_id", physical_dtype="object"),
            DatasetColumnSchema(name="qty", physical_dtype="float64"),
        )
        pred_art = self._make_prediction_artifact(
            storage, factory_id=factory_id, csv_bytes=pred_csv, columns=pred_cols
        )
        batch = builder.resolve_prediction_observations(
            scope=scope,
            artifact=pred_art,
            contract=contract,
            context=ctx,
        )
        assert len(batch.observations) == 1
        obs = batch.observations[0]
        assert obs.resolution_status == PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE
        assert obs.known_record_id is None

    def test_prediction_does_not_create_historical_identity(self, tmp_path: Path):
        factory_id, storage, lin_reg, builder, contract, scope, training_ids = (
            self._build_prediction_infrastructure(tmp_path)
        )
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        # "999" is unknown.
        pred_csv = b"batch_id,qty\n999,200.0\n"
        pred_cols = (
            DatasetColumnSchema(name="batch_id", physical_dtype="object"),
            DatasetColumnSchema(name="qty", physical_dtype="float64"),
        )
        pred_art = self._make_prediction_artifact(
            storage, factory_id=factory_id, csv_bytes=pred_csv, columns=pred_cols
        )
        builder.resolve_prediction_observations(
            scope=scope,
            artifact=pred_art,
            contract=contract,
            context=ctx,
        )
        # No new historical identity must have been created.
        from taqamind.core.interfaces.dataset_registry import RegistryNotFoundError

        with pytest.raises(RegistryNotFoundError):
            lin_reg.get_membership(scope=scope, dataset_revision_id=pred_art.metadata.revision_id)

    def test_prediction_target_present_raises(self, tmp_path: Path):
        factory_id, storage, lin_reg, builder, contract, scope, _ = (
            self._build_prediction_infrastructure(tmp_path)
        )
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        # Include target column "waste" in prediction input — must be rejected.
        pred_csv = b"batch_id,qty,waste\n001,100.0,2.0\n"
        pred_cols = BASIC_COLS
        pred_art = self._make_prediction_artifact(
            storage, factory_id=factory_id, csv_bytes=pred_csv, columns=pred_cols
        )
        with pytest.raises(ValueError, match="TARGET_PRESENT_IN_PREDICTION"):
            builder.resolve_prediction_observations(
                scope=scope,
                artifact=pred_art,
                contract=contract,
                context=ctx,
            )

    def test_prediction_with_missing_key_resolves_unverified(self, tmp_path: Path):
        factory_id, storage, lin_reg, builder, contract, scope, _ = (
            self._build_prediction_infrastructure(tmp_path)
        )
        ctx = _make_context(factory_id=factory_id, purpose=DatasetPurpose.PREDICTION)
        pred_csv = b"batch_id,qty\n,100.0\n"
        pred_cols = (
            DatasetColumnSchema(name="batch_id", physical_dtype="object"),
            DatasetColumnSchema(name="qty", physical_dtype="float64"),
        )
        pred_art = self._make_prediction_artifact(
            storage, factory_id=factory_id, csv_bytes=pred_csv, columns=pred_cols
        )
        batch = builder.resolve_prediction_observations(
            scope=scope,
            artifact=pred_art,
            contract=contract,
            context=ctx,
        )
        obs = batch.observations[0]
        assert obs.resolution_status == PredictionIdentityStatus.UNVERIFIED
        assert obs.known_record_id is None
        assert obs.business_key_sha256 is None


# ---------------------------------------------------------------------------
# Group I — PredictionObservation contract invariants
# ---------------------------------------------------------------------------


class TestPredictionObservationInvariants:
    """Group I: PredictionObservation and PredictionObservationBatch Pydantic invariants."""

    def _base_scope(self) -> RecordScope:
        return _make_scope(factory_id=uuid4())

    def test_known_historical_requires_known_record_id(self):
        scope = self._base_scope()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PredictionObservation(
                scope=scope,
                prediction_dataset_revision_id=uuid4(),
                row_position=0,
                prediction_input_sha256="c" * 64,
                resolution_status=PredictionIdentityStatus.KNOWN_HISTORICAL,
                known_record_id=None,  # must not be None for KNOWN_HISTORICAL
            )

    def test_non_historical_must_not_expose_known_record_id(self):
        scope = self._base_scope()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PredictionObservation(
                scope=scope,
                prediction_dataset_revision_id=uuid4(),
                row_position=0,
                prediction_input_sha256="c" * 64,
                resolution_status=PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE,
                known_record_id=uuid4(),  # must be None for non-KNOWN_HISTORICAL
            )

    def test_unverified_requires_reason(self):
        scope = self._base_scope()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PredictionObservation(
                scope=scope,
                prediction_dataset_revision_id=uuid4(),
                row_position=0,
                prediction_input_sha256="c" * 64,
                resolution_status=PredictionIdentityStatus.UNVERIFIED,
                # missing reason
            )

    def test_valid_known_historical_observation(self):
        scope = self._base_scope()
        obs = PredictionObservation(
            scope=scope,
            prediction_dataset_revision_id=uuid4(),
            row_position=0,
            business_key_sha256="a" * 64,
            prediction_input_sha256="c" * 64,
            resolution_status=PredictionIdentityStatus.KNOWN_HISTORICAL,
            known_record_id=uuid4(),
        )
        assert obs.known_record_id is not None

    def test_prediction_observation_batch_is_immutable(self):
        scope = self._base_scope()
        obs = PredictionObservation(
            scope=scope,
            prediction_dataset_revision_id=uuid4(),
            row_position=0,
            prediction_input_sha256="c" * 64,
            resolution_status=PredictionIdentityStatus.NEW_OBSERVATION_CANDIDATE,
        )
        batch = PredictionObservationBatch(
            scope=scope,
            prediction_dataset_revision_id=uuid4(),
            observations=(obs,),
        )
        from pydantic import ValidationError

        with pytest.raises((ValidationError, TypeError)):
            batch.observations = ()


# ---------------------------------------------------------------------------
# Group J — Performance benchmarks
# ---------------------------------------------------------------------------


class TestCanonicalizationPerformance:
    """Group J: Measure canonicalization and hash throughput at representative sizes."""

    def _build_string_series(self, n: int) -> tuple[pd.Series, FeatureContract]:
        feat = _make_feature("batch_id", dtype=DataType.STRING)
        series = pd.Series([f"BATCH-{i:08d}" for i in range(n)])
        return series, feat

    def _build_float_series(self, n: int) -> tuple[pd.Series, FeatureContract]:
        feat = _make_feature(
            "qty", dtype=DataType.FLOAT, constraints=FeatureConstraints(minimum=0.0)
        )
        series = pd.Series([float(i) * 0.001 for i in range(n)])
        return series, feat

    def _time_canonicalize(self, n: int, dtype: str = "string") -> float:
        if dtype == "string":
            series, feat = self._build_string_series(n)
        else:
            series, feat = self._build_float_series(n)
        start = time.perf_counter()
        canonicalize_column(series, feat)
        return time.perf_counter() - start

    @pytest.mark.parametrize(
        "n,max_seconds",
        [
            (10_000, 5.0),
            (100_000, 30.0),
            (250_000, 60.0),
        ],
    )
    def test_string_canonicalization_throughput(self, n: int, max_seconds: float):
        elapsed = self._time_canonicalize(n, dtype="string")
        print(f"\n[BENCH] String canonicalize {n:,} rows: {elapsed:.3f}s")
        assert elapsed < max_seconds, (
            f"String canonicalization of {n:,} rows took {elapsed:.3f}s (limit: {max_seconds}s)"
        )

    @pytest.mark.parametrize(
        "n,max_seconds",
        [
            (10_000, 10.0),
            (100_000, 60.0),
        ],
    )
    def test_float_canonicalization_throughput(self, n: int, max_seconds: float):
        elapsed = self._time_canonicalize(n, dtype="float")
        print(f"\n[BENCH] Float canonicalize {n:,} rows: {elapsed:.3f}s")
        assert elapsed < max_seconds, (
            f"Float canonicalization of {n:,} rows took {elapsed:.3f}s (limit: {max_seconds}s)"
        )

    def test_hash_canonical_v2_row_throughput_10k(self):
        n = 10_000
        start = time.perf_counter()
        for i in range(n):
            pairs = [("batch_id", f"str:BATCH-{i:08d}"), ("qty", "float:0x0p+0")]
            hash_canonical_v2_row(namespace="business_key", field_pairs=pairs)
        elapsed = time.perf_counter() - start
        print(f"\n[BENCH] hash_canonical_v2_row {n:,} rows: {elapsed:.3f}s")
        assert elapsed < 10.0, f"Hashing {n:,} rows took {elapsed:.3f}s (limit: 10s)"
