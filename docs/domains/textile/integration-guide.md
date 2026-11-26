# Integration Guide

## Domain Registration

```python
from taqamind.core.registry import DomainRegistry
from taqamind.domains.textile import register_textile

registry = DomainRegistry()
register_textile(registry)  # TargetConfirmation defaults to UNCONFIRMED
```

## Service Usage

```python
from uuid import uuid4
from taqamind.core.contracts import DatasetPurpose
from taqamind.domains.textile import TextilePipelineService

service = TextilePipelineService("artifacts/textile/store")
factory_id = uuid4()
# 1. Prepare data -> 2. Train -> 3. Predict
```

## CLI Entrypoint

```powershell
python -m taqamind.domains.textile.cli --help
python -m taqamind.domains.textile.cli inspect --process weaving
python -m taqamind.domains.textile.cli template --output-dir contracts/domains/textile/templates
```
