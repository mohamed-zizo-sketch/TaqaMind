"""Exceptions raised by the persistence layer."""


class PersistenceError(Exception):
    """Base exception for persistence-related failures."""


class DatabaseConfigurationError(PersistenceError):
    """Raised when database configuration is invalid."""


class DatabaseConnectionError(PersistenceError):
    """Raised when a database connection cannot be established."""


class DatabaseTransactionError(PersistenceError):
    """Raised when a database transaction cannot be completed."""
