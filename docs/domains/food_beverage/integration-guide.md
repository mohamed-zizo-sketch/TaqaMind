# TaqaMind — Food & Beverage Integration Guide

---

## 1. Domain Registration

```python
from taqamind.core.registry import domain_registry
from taqamind.domains.food_beverage import register_food_beverage

domain = register_food_beverage(domain_registry)
```

## 2. Programmatic Entrypoints

```python
from pathlib import Path
from uuid import uuid4
from taqamind.domains.food_beverage import (
    FoodBeveragePipeline,
    FoodBeverageStorageConfig,
    make_synthetic_target_confirmation,
)

pipe = FoodBeveragePipeline(storage_config=FoodBeverageStorageConfig(root_dir=Path("artifacts/food_beverage")))
factory_id = uuid4()

# Train (requires confirmed TargetConfirmation)
train_res = pipe.train_model(
    csv_path=Path("train.csv"),
    factory_id=factory_id,
    process_id="bakery",
    target_confirmation=make_synthetic_target_confirmation(),
)

# Predict (inference-only, never retrains)
pred_batch = pipe.predict(
    csv_path=Path("predict.csv"),
    factory_id=factory_id,
    process_id="bakery",
)
```

## 3. CLI Entrypoints

```bash
.venv/Scripts/python.exe -m taqamind.domains.food_beverage.cli --help
.venv/Scripts/python.exe -m taqamind.domains.food_beverage.cli template --process bakery --purpose training --output train_tpl.csv
.venv/Scripts/python.exe -m taqamind.domains.food_beverage.cli train --csv train.csv --factory-id <UUID> --process bakery --synthetic-confirm
.venv/Scripts/python.exe -m taqamind.domains.food_beverage.cli predict --csv predict.csv --factory-id <UUID> --process bakery
.venv/Scripts/python.exe -m taqamind.domains.food_beverage.cli model-info --factory-id <UUID> --process bakery
```

## 4. Storage Layout

- `artifacts/food_beverage/datasets/raw/<factory_id>/food_beverage/<process>/`
- `artifacts/food_beverage/datasets/processed/<factory_id>/food_beverage/<process>/`
- `artifacts/food_beverage/models/<factory_id>/food_beverage/<process>/<artifact_id>.joblib`
- `artifacts/food_beverage/registries/model_registry.json`

## 5. Replacing Local Adapters with Future Shared Core Engines

`FoodBeveragePipeline` instantiates all engines in `__init__`. Once Shared Core implements `taqamind.core.ingestion`, `taqamind.core.mapping`, etc., swap the adapter imports in `src/taqamind/domains/food_beverage/pipeline.py` with the Core concrete classes — zero changes to domain contracts or CLI signatures are required.
