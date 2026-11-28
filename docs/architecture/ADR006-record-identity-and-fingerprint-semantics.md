# ADR006 — Record Identity and Fingerprint Semantics

- **Status:** Accepted
- **Date:** 2026-11-28
- **Work Package:** WP02 (REV006, REV019, REV020, REV022, REV023)

## Context

TaqaMind processes industrial CSV datasets across multiple factories, domains, processes, and dataset revisions. During V1 audit, five defects were identified in record identity and fingerprinting:

1. **REV006:** Inconsistent business key whitespace normalization across semantic validation, deterministic cleaning, and lineage membership construction.
2. **REV019:** Lossy `round(float(value), 9)` float canonicalization and untyped JSON serialization causing precision loss and cross-type collisions (`1` vs `1.0` vs `"1"` vs `True`).
3. **REV020:** Failed `ContractTypeConverter` conversions silently ignored in `RecordMembershipBuilder`, leaving raw invalid tokens in fingerprint inputs.
4. **REV022:** Incomplete scope, contract, and row-level preflight validation before mutating `RecordLineageRegistry`.
5. **REV023:** Prediction datasets routed through historical `RecordMembershipBuilder`, confusing prediction observations with confirmed historical ground-truth versions.

## Decision

### 1. Four Distinct Fingerprint Responsibilities

TaqaMind separates four distinct cryptographic responsibilities:

1. **Raw File Fingerprint (`content_sha256` on `ArtifactReference`):**
   - SHA-256 of the exact immutable uploaded bytes. Used for byte integrity verification, never as business record identity.
2. **Business Key Fingerprint (`business_key_sha256`, namespace `business_key`):**
   - Identifies a business record within its authoritative `RecordScope` (`factory_id`, `domain`, `process`, `dataset_id`, `identity_namespace`, `identity_version`).
3. **Historical Content Fingerprint (`content_sha256` on `RecordVersion`, namespace `record_content`, version `2.0.0`):**
   - Identifies the canonical observed content version of a historical production record.
   - Excludes recomputable `FeatureOrigin.DERIVED` fields so feature engineering implementation changes do not trigger false historical record corrections.
   - Includes authoritative business keys, recorded/measured non-derived fields, ground-truth target (when present in non-prediction datasets), and canonical schema provenance.
4. **Prediction Input Fingerprint (`prediction_input_sha256`, namespace `prediction_input`, version `2.0.0`):**
   - Identifies validated pre-production prediction inputs for an inference observation without treating them as historical ground-truth content.
   - Excludes ground-truth targets, model outputs, runtime prediction timestamps, random IDs, upload IDs, undeclared columns, and post-production fields.
   - Not a complete prediction cache key (WP08 will bind model version and execution configuration).

### 2. Authorization & Scope Isolation

- A fingerprint is a deterministic content/identity digest, **not** an authorization or access-control mechanism.
- All identity and observation operations enforce `factory_id`, `domain`, `process`, `dataset_id`, `identity_namespace`, and `identity_version` isolation.

### 3. Contract-Driven Canonicalization (`v2.0.0`)

- **STRING:** Trim surrounding whitespace; preserve case, internal whitespace, leading zeros, and valid literal `"NA"`/`"NULL"`/`"NaN"` tokens.
- **CATEGORY:** Trim surrounding whitespace; enforce declared `allowed_values`.
- **INTEGER:** Exact signed 64-bit `Int64` string representation; never routed through `float64`.
- **FLOAT:** Exact binary64 representation via `float.hex()` with `-0.0` normalized to `+0.0` and non-finite values rejected. Preserves validated `Float64` precision (not arbitrary-precision decimal source).
- **BOOLEAN:** Actual Python `bool` via `ContractTypeConverter` (never `bool(raw_string)`).
- **DATE / DATETIME:** Validated ISO 8601 date and UTC nanosecond-precision datetime.
- **Type-Tagged Serialization:** Every canonical field encodes `{"t": "<type_tag>", "v": <canonical_value>}` to prevent cross-type collisions (`"1"` vs `1` vs `1.0` vs `True` vs `None` vs `"null"`).

### 4. Legacy Compatibility & Non-Destructive Migration

- `RecordScope.identity_version` is part of the filesystem registry path and is **not** automatically incremented when fingerprint serialization changes.
- Legacy `1.0.0` hashes, `RecordIdentity` entries, `RecordVersion` entries, `DatasetMembershipManifest` records, and exposure history remain untouched.
- For `1.0.0` -> `2.0.0` version mismatches, `RecordLineageRegistry.match` returns `RecordMatchStatus.MIGRATION_REQUIRED`, which cannot be bypassed by `confirm_updates_by`.
- Lossy legacy `FLOAT` business keys fail closed when exact identity equivalence cannot be proven.

### 5. Prediction Observation Separation

- `DatasetPurpose.PREDICTION` is prohibited from creating `RecordIdentity`, `RecordVersion`, or `DatasetMembershipManifest` entries in historical lineage.
- `CorePredictionEngine` resolves prediction rows read-only into `PredictionObservation` contracts (`KNOWN_HISTORICAL` with `known_record_id`, or `NEW_OBSERVATION_CANDIDATE` / `UNVERIFIED` / `AMBIGUOUS` / `MIGRATION_REQUIRED` with `known_record_id=None`).

## Consequences

- Shared Core callers (`CorePredictionEngine`, E2E tests) no longer register prediction datasets into historical lineage.
- Domain pipelines (`textile`, `iron_manufacturing`, `food_beverage`) that still call historical membership builders on prediction datasets or rely on `1.0.0` adapters are documented for WP10 migration.
- Transactional all-or-nothing persistence and persistent prediction observation storage remain deferred to WP03.
