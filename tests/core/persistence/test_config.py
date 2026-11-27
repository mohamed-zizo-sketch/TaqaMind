"""Unit tests for PostgreSQL persistence configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from taqamind.core.persistence.config import DatabaseConfig, DatabaseRole
from taqamind.core.persistence.exceptions import DatabaseConfigurationError

ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "DB_APP_USER",
    "DB_APP_PASSWORD",
    "DB_MIGRATOR_USER",
    "DB_MIGRATOR_PASSWORD",
    "DB_SCHEMA",
    "DB_POOL_SIZE",
    "DB_MAX_OVERFLOW",
    "DB_POOL_TIMEOUT",
    "DB_POOL_RECYCLE",
    "DB_CONNECT_TIMEOUT",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real environment variables from affecting unit tests."""

    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def valid_settings() -> dict:
    return {
        "host": "127.0.0.1",
        "port": 5433,
        "database": "taqamind",
        "username": "taqamind_app",
        "password": "test-password-only",
    }


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"

    path.write_text(
        "\n".join(
            [
                "POSTGRES_HOST=127.0.0.1",
                "POSTGRES_PORT=5433",
                "POSTGRES_DB=taqamind",
                "DB_APP_USER=taqamind_app",
                "DB_APP_PASSWORD=app-test-password",
                "DB_MIGRATOR_USER=taqamind_migrator",
                "DB_MIGRATOR_PASSWORD=migrator-test-password",
                "DB_SCHEMA=taqamind",
                "DB_POOL_SIZE=5",
                "DB_MAX_OVERFLOW=10",
                "DB_POOL_TIMEOUT=30",
                "DB_POOL_RECYCLE=1800",
                "DB_CONNECT_TIMEOUT=10",
            ]
        ),
        encoding="utf-8",
    )

    return path


def test_valid_configuration(valid_settings: dict) -> None:
    config = DatabaseConfig(**valid_settings)

    assert config.host == "127.0.0.1"
    assert config.port == 5433
    assert config.database == "taqamind"
    assert config.role == DatabaseRole.APPLICATION
    assert config.schema_name == "taqamind"


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_invalid_port(
    valid_settings: dict,
    port: int,
) -> None:
    valid_settings["port"] = port

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


@pytest.mark.parametrize("password", ["", "   ", "CHANGE_ME"])
def test_invalid_password(
    valid_settings: dict,
    password: str,
) -> None:
    valid_settings["password"] = password

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


@pytest.mark.parametrize(
    "schema_name",
    ["", "Invalid Schema", "123schema", "schema;DROP", "a" * 64],
)
def test_invalid_schema(
    valid_settings: dict,
    schema_name: str,
) -> None:
    valid_settings["schema_name"] = schema_name

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pool_size", 0),
        ("max_overflow", -1),
        ("pool_timeout", 0),
        ("pool_recycle", 0),
        ("connect_timeout", 0),
    ],
)
def test_invalid_pool_settings(
    valid_settings: dict,
    field: str,
    value: int,
) -> None:
    valid_settings[field] = value

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


@pytest.mark.parametrize(
    "field",
    ["host", "database", "username"],
)
def test_blank_identifiers(
    valid_settings: dict,
    field: str,
) -> None:
    valid_settings[field] = "   "

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


def test_unknown_fields_rejected(valid_settings: dict) -> None:
    valid_settings["unknown_setting"] = True

    with pytest.raises(ValidationError):
        DatabaseConfig(**valid_settings)


def test_configuration_is_immutable(valid_settings: dict) -> None:
    config = DatabaseConfig(**valid_settings)

    with pytest.raises(ValidationError):
        config.port = 5432


def test_password_is_masked(valid_settings: dict) -> None:
    config = DatabaseConfig(**valid_settings)
    password = valid_settings["password"]

    assert password not in repr(config)
    assert password not in str(config.password)
    assert password not in str(config.to_url())
    assert password not in config.model_dump_json()


def test_special_password_characters(valid_settings: dict) -> None:
    password = "test@password:#/%?+"

    valid_settings["password"] = password

    config = DatabaseConfig(**valid_settings)
    url = config.to_url()

    assert url.password == password
    assert url.drivername == "postgresql+psycopg"
    assert password not in str(url)


def test_load_application_configuration(env_file: Path) -> None:
    config = DatabaseConfig.from_env(
        role=DatabaseRole.APPLICATION,
        env_file=env_file,
    )

    assert config.username == "taqamind_app"
    assert config.password.get_secret_value() == "app-test-password"
    assert config.port == 5433


def test_load_migrator_configuration(env_file: Path) -> None:
    config = DatabaseConfig.from_env(
        role=DatabaseRole.MIGRATOR,
        env_file=env_file,
    )

    assert config.username == "taqamind_migrator"
    assert config.password.get_secret_value() == "migrator-test-password"


def test_environment_overrides_file(
    env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "5434")
    monkeypatch.setenv("DB_POOL_SIZE", "8")

    config = DatabaseConfig.from_env(env_file=env_file)

    assert config.port == 5434
    assert config.pool_size == 8


def test_missing_environment_file(tmp_path: Path) -> None:
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_env(
            env_file=tmp_path / "missing.env",
        )


def test_missing_credentials() -> None:
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_env()


def test_invalid_environment_values(
    env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "invalid-port")

    with pytest.raises(
        DatabaseConfigurationError,
        match="port",
    ):
        DatabaseConfig.from_env(env_file=env_file)


def test_configuration_error_does_not_expose_password(
    env_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "sensitive-test-password"

    monkeypatch.setenv("DB_APP_PASSWORD", password)
    monkeypatch.setenv("POSTGRES_PORT", "invalid")

    with pytest.raises(DatabaseConfigurationError) as exc:
        DatabaseConfig.from_env(env_file=env_file)

    assert password not in str(exc.value)


def test_invalid_database_role(env_file: Path) -> None:
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_env(
            role="admin",
            env_file=env_file,
        )


def test_loading_does_not_modify_environment(
    env_file: Path,
) -> None:
    import os

    before = dict(os.environ)

    DatabaseConfig.from_env(env_file=env_file)

    assert dict(os.environ) == before
