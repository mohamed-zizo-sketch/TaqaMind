# Industrial Domain Registration

## Purpose

TaqaMind uses a shared industrial data and machine learning architecture across multiple industrial domains.

Each domain defines its own industrial processes, tasks, dataset contracts, features, targets and business rules while reusing the shared Core interfaces and implementations.

## Initial domains

| Domain ID          | Responsibility                   |
| ------------------ | -------------------------------- |
| textile            | Textile industry team            |
| food_beverage      | Food and beverage industry team  |
| iron_manufacturing | Iron manufacturing industry team |

## Registration hierarchy

DomainContract → ProcessContract → IndustrialTaskContract → DatasetContract

Each domain must provide validated and versioned contracts before its pipeline can be used for production training.

## Domain team responsibilities

- Research industrial processes and dataset sources.
- Establish the correct record granularity and business keys.
- Define features, units and feature availability.
- Define prediction targets and measurement semantics.
- Implement domain-specific transformations and validation rules.
- Provide synthetic tests and dataset research reports.
- Integrate existing pipelines with the shared Core interfaces.

## Shared Core responsibilities

- Dataset ingestion and profiling.
- Column mapping and semantic validation.
- Dataset and record lineage.
- Common preprocessing mechanisms.
- Training readiness and leakage prevention.
- Controlled model training and evaluation.
- Model versioning and prediction.
- Storage and registry services.

## Mandatory rules

1. Do not duplicate shared Core functionality inside domain packages.
2. Do not import another team's domain-specific implementation directly.
3. Propose reusable components for shared Core review.
4. Do not assume that external research datasets represent Egyptian factories.
5. Do not include post-production predictors in pre-production prediction tasks.
6. Do not retrain a model automatically when prediction data is uploaded.
7. Do not commit confidential factory datasets or trained model artifacts.
8. Register domain versions explicitly.
9. Do not treat unconfirmed industrial assumptions as verified contracts.

## Current implementation status

The shared contracts and interfaces define the intended architecture. Concrete storage services, processing engines and machine learning implementations are still under development.

The Domain Registry currently provides in-memory registration and discovery. Registration does not establish production training readiness or factory approval.
