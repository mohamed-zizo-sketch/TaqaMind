"""Validated PostgreSQL configuration for the shared persistence layer."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from sqlalchemy.engine import URL

from taqamind.core.persistence.exceptions import DatabaseConfigurationError


class DatabaseRole(StrEnum):
    """Supported database connection roles."""

    APPLICATION = "application"
    MIGRATOR = "migrator"


class DatabaseConfig(BaseModel):
    """Immutable and validated PostgreSQL connection settings."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    role: DatabaseRole = DatabaseRole.APPLICATION

    host: str = Field(min_length=1)
    port: int = Field(default=5432, ge=1, le=65535)

    database: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: SecretStr

    schema_name: str = Field(
        default="taqamind",
        pattern=r"^[a-z_][a-z0-9_]{0,62}$",
    )

    pool_size: int = Field(default=5, ge=1)
    max_overflow: int = Field(default=10, ge=0)

    pool_timeout: int = Field(default=30, gt=0)
    pool_recycle: int = Field(default=1800, gt=0)
    connect_timeout: int = Field(default=10, gt=0)

    @field_validator("host", "database", "username")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        """Reject blank connection identifiers."""

        value = value.strip()

        if not value:
            raise ValueError("Database identifier cannot be blank.")

        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        """Reject empty and placeholder credentials."""

        secret = value.get_secret_value()

        if not secret.strip():
            raise ValueError("Database password cannot be empty.")

        if secret == "CHANGE_ME":
            raise ValueError("Database password must be configured.")

        return value

    def to_url(self) -> URL:
        """Build a PostgreSQL URL without manual string interpolation."""

        return URL.create(
            drivername="postgresql+psycopg",
            username=self.username,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.database,
        )

    @classmethod
    def from_env(
        cls,
        *,
        role: DatabaseRole = DatabaseRole.APPLICATION,
        env_file: str | Path | None = None,
    ) -> DatabaseConfig:
        """Load settings explicitly from environment and an optional file."""

        settings: dict[str, str | None] = {}

        if env_file is not None:
            path = Path(env_file)

            if not path.is_file():
                raise DatabaseConfigurationError("The specified environment file does not exist.")

            try:
                settings.update(
                    dotenv_values(
                        dotenv_path=path,
                        interpolate=False,
                    )
                )
            except (OSError, UnicodeError):
                raise DatabaseConfigurationError("Unable to read the environment file.") from None

        # Real environment variables take precedence over local files.
        settings.update(os.environ)

        try:
            role = DatabaseRole(role)
        except ValueError:
            raise DatabaseConfigurationError("Unsupported database role.") from None

        prefix = "DB_APP" if role == DatabaseRole.APPLICATION else "DB_MIGRATOR"

        try:
            return cls(
                role=role,
                host=settings.get("POSTGRES_HOST", "127.0.0.1"),
                port=settings.get("POSTGRES_PORT", "5432"),
                database=settings.get("POSTGRES_DB"),
                username=settings.get(f"{prefix}_USER"),
                password=settings.get(f"{prefix}_PASSWORD"),
                schema_name=settings.get("DB_SCHEMA", "taqamind"),
                pool_size=settings.get("DB_POOL_SIZE", "5"),
                max_overflow=settings.get("DB_MAX_OVERFLOW", "10"),
                pool_timeout=settings.get("DB_POOL_TIMEOUT", "30"),
                pool_recycle=settings.get("DB_POOL_RECYCLE", "1800"),
                connect_timeout=settings.get("DB_CONNECT_TIMEOUT", "10"),
            )

        except ValidationError as exc:
            invalid_fields = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors(include_input=False)
                }
            )

            raise DatabaseConfigurationError(
                "Invalid database configuration fields: " + ", ".join(invalid_fields)
            ) from None
