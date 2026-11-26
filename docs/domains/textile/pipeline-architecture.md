# Textile Pipeline Architecture

## End-to-End Flow

```
RAW CSV
  └─▶ CsvDataIngestion (Core P1.2) ──────────────────────▶ INGESTED
        └─▶ UnitAwareColumnMapperAdapter (AUD-001) ──────▶ MAPPED
              └─▶ ContractSemanticValidator (Core P1.3)
                    └─▶ ContractDataCleaner (Core P1.5) ─▶ CLEANED
                          └─▶ DeterministicFeatureEngineer (Core P1.5) ─▶ ENGINEERED
                                └─▶ TextileRowMembershipBuilder (AUD-008) ─▶ RecordLineage
                                      │
                                      ├─[TRAINING]──▶ TextileTrainingReadinessChecker
                                      │                 └─▶ TextileDatasetSplitter (GROUP / GROUP_TEMPORAL)
                                      │                       └─▶ TextileLeakageValidator
                                      │                             └─▶ TextileControlledAutoMLTrainer (train/val only)
                                      │                                   └─▶ TextileProtectedFinalEvaluator (test once)
                                      │                                         └─▶ LocalModelRegistry
                                      │
                                      └─[PREDICTION]─▶ LocalActiveModelProvider (verified SHA-256)
                                                        └─▶ TextilePredictionEngine (transform-only, ZERO fit)
```

## Module Layout

- `src/taqamind/domains/textile/common/`: SI unit conversions, Excel yarn-count recovery, engineering formulas.
- `src/taqamind/domains/textile/weaving/`: Weaving `DatasetContract`, `ProcessContract`, derived specs, synthetic generators.
- `src/taqamind/domains/textile/knitting/`: Knitting `DatasetContract`, `ProcessContract`, derived specs, synthetic generators.
- `src/taqamind/domains/textile/adapters/`: Replaceable P1.6–P1.8 and AUD-001/AUD-008 adapters.
- `src/taqamind/domains/textile/pipeline.py`: `TextilePipelineService` and `TextilePlatformStores`.
- `src/taqamind/domains/textile/cli.py`: CLI (`inspect`, `template`, `validate`, `train`, `retrain`, `predict`, `model-info`).
