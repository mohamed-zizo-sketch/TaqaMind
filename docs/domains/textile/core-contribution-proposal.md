# Core Contribution Proposal (Adapter -> Shared Core Migration)

All temporary adapters in `src/taqamind/domains/textile/adapters/` implement Core protocols and can be retired cleanly once Core P1.6–P1.8 and audit fixes land:

| Textile Adapter | Target Core Module | Reason for Adapter |
|---|---|---|
| `UnitAwareColumnMapperAdapter` | `taqamind.core.mapping.DeterministicColumnMapper` | Mitigates AUD-001 (missing unit conversion in `apply_mapping`). |
| `TextileRowMembershipBuilder` | `taqamind.core.lineage` | Mitigates AUD-008 (missing row-to-record membership builder). |
| `TextileTrainingReadinessChecker`, `TextileDatasetSplitter`, `TextileLeakageValidator`, `LocalSplitRegistry`, `LocalRecordUsageRegistry` | `taqamind.core.readiness`, `taqamind.core.splitting` (P1.6) | Core P1.6 concrete implementations not yet implemented. |
| `LocalDevelopmentDataProvider`, `LocalProtectedTestDataProvider`, `LocalTrainingRegistry`, `LocalEvaluationRegistry`, `TextileControlledAutoMLTrainer`, `TextileProtectedFinalEvaluator` | `taqamind.core.automl`, `taqamind.core.evaluation` (P1.7) | Core P1.7 concrete implementations not yet implemented. |
| `LocalModelRegistry`, `LocalActiveModelProvider`, `TextilePredictionEngine` | `taqamind.core.prediction`, `taqamind.core.registry` (P1.8) | Core P1.8 concrete implementations not yet implemented. |
