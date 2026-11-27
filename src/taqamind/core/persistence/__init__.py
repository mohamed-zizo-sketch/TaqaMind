"""Shared PostgreSQL persistence infrastructure."""

from taqamind.core.persistence.config import (
    DatabaseConfig,
    DatabaseRole,
)
from taqamind.core.persistence.database import (
    Database,
    DatabaseHealth,
)
from taqamind.core.persistence.exceptions import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseTransactionError,
    PersistenceError,
)

__all__ = [
    "Database",
    "DatabaseConfig",
    "DatabaseConfigurationError",
    "DatabaseConnectionError",
    "DatabaseHealth",
    "DatabaseRole",
    "DatabaseTransactionError",
    "PersistenceError",
]
