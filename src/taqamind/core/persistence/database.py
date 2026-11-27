"""PostgreSQL engine and transaction management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from taqamind.core.persistence.config import DatabaseConfig
from taqamind.core.persistence.exceptions import (
    DatabaseConnectionError,
    DatabaseTransactionError,
)


@dataclass(frozen=True, slots=True)
class DatabaseHealth:
    """Verified database connection information."""

    database: str
    username: str
    schema_name: str
    server_version: str


class Database:
    """Manage PostgreSQL connections and explicit transaction boundaries."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config

        self._engine = create_engine(
            config.to_url(),
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=True,
            pool_reset_on_return="rollback",
            isolation_level="READ COMMITTED",
            connect_args={
                "connect_timeout": config.connect_timeout,
                "application_name": f"taqamind-{config.role.value}",
                "options": f"-csearch_path={config.schema_name}",
            },
            echo=False,
            hide_parameters=True,
        )

    @property
    def config(self) -> DatabaseConfig:
        """Return immutable database configuration."""
        return self._config

    @property
    def engine(self) -> Engine:
        """Expose the engine for infrastructure integrations."""
        return self._engine

    def check_connection(self) -> DatabaseHealth:
        """Verify connectivity and expected database identity."""

        try:
            with self._engine.connect() as connection:
                row = (
                    connection.execute(
                        text(
                            """
                        SELECT
                            current_database() AS database,
                            current_user AS username,
                            current_schema() AS schema_name,
                            current_setting('server_version')
                                AS server_version
                        """
                        )
                    )
                    .mappings()
                    .one()
                )

        except SQLAlchemyError:
            raise DatabaseConnectionError(
                "Unable to establish a valid PostgreSQL connection."
            ) from None

        if row["database"] != self._config.database:
            raise DatabaseConnectionError("Connected to an unexpected database.")

        if row["username"] != self._config.username:
            raise DatabaseConnectionError("Connected using an unexpected database role.")

        if row["schema_name"] != self._config.schema_name:
            raise DatabaseConnectionError("Connected to an unexpected database schema.")

        return DatabaseHealth(
            database=row["database"],
            username=row["username"],
            schema_name=row["schema_name"],
            server_version=row["server_version"],
        )

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Execute operations inside an atomic write transaction."""

        try:
            with self._engine.begin() as connection:
                yield connection

        except SQLAlchemyError:
            raise DatabaseTransactionError("Database transaction failed.") from None

    @contextmanager
    def read_transaction(self) -> Iterator[Connection]:
        """Execute operations inside a PostgreSQL read-only transaction."""

        try:
            with self._engine.begin() as connection:
                connection.execute(text("SET TRANSACTION READ ONLY"))
                yield connection

        except SQLAlchemyError:
            raise DatabaseTransactionError("Read-only database transaction failed.") from None

    def dispose(self) -> None:
        """Release pooled database connections."""
        self._engine.dispose()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.dispose()
