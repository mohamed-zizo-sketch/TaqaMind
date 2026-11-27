"""PostgreSQL persistence integration tests."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

from taqamind.core.persistence import (
    Database,
    DatabaseConfig,
    DatabaseRole,
    DatabaseTransactionError,
)

TEST_DATABASE = "taqamind_test"

TEST_USERS = {
    DatabaseRole.APPLICATION: "taqamind_test_app",
    DatabaseRole.MIGRATOR: "taqamind_test_migrator",
}

ENV_FILE = Path(".env.test")

pytestmark = pytest.mark.skipif(
    os.environ.get("TAQAMIND_RUN_DB_TESTS") != "1",
    reason="PostgreSQL integration tests require explicit opt-in.",
)


@pytest.fixture(scope="module")
def databases():
    """Create isolated application and migration engines."""

    instances = {}

    try:
        for role, expected_user in TEST_USERS.items():
            config = DatabaseConfig.from_env(
                role=role,
                env_file=ENV_FILE,
            )

            # Never run integration tests against the main database.
            if config.database != TEST_DATABASE:
                raise RuntimeError("Unsafe testing database name.")

            if config.username != expected_user:
                raise RuntimeError("Unsafe testing database credentials.")

            if config.schema_name != "taqamind":
                raise RuntimeError("Unexpected testing schema.")

            database = Database(config)
            instances[role] = database

            database.check_connection()

        yield instances

    finally:
        for database in instances.values():
            database.dispose()


@pytest.fixture
def app_db(databases) -> Database:
    return databases[DatabaseRole.APPLICATION]


@pytest.fixture
def migrator_db(databases) -> Database:
    return databases[DatabaseRole.MIGRATOR]


@pytest.fixture
def test_table(migrator_db: Database):
    """Create an isolated table and remove it after each test."""

    table = f"taqamind.integration_{uuid4().hex}"

    with migrator_db.transaction() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE {table} (
                    id INTEGER PRIMARY KEY,
                    value INTEGER NOT NULL
                )
                """
            )
        )

        conn.execute(
            text(
                f"""
                GRANT SELECT, INSERT, UPDATE, DELETE
                ON TABLE {table}
                TO taqamind_test_app
                """
            )
        )

    try:
        yield table

    finally:
        with migrator_db.transaction() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def count_records(database: Database, table: str) -> int:
    """Return the number of committed records."""

    with database.read_transaction() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()


def test_application_connection(app_db: Database) -> None:
    health = app_db.check_connection()

    assert health.database == TEST_DATABASE
    assert health.username == "taqamind_test_app"
    assert health.schema_name == "taqamind"


def test_migrator_connection(migrator_db: Database) -> None:
    health = migrator_db.check_connection()

    assert health.database == TEST_DATABASE
    assert health.username == "taqamind_test_migrator"
    assert health.schema_name == "taqamind"


def test_transaction_commit(
    app_db: Database,
    test_table: str,
) -> None:
    with app_db.transaction() as conn:
        conn.execute(
            text(f"INSERT INTO {test_table} VALUES (:id, :value)"),
            {"id": 1, "value": 100},
        )

    assert count_records(app_db, test_table) == 1


def test_transaction_rollback(
    app_db: Database,
    test_table: str,
) -> None:
    with pytest.raises(RuntimeError, match="Simulated failure"):
        with app_db.transaction() as conn:
            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 100)"))

            raise RuntimeError("Simulated failure")

    assert count_records(app_db, test_table) == 0


def test_transaction_atomicity(
    app_db: Database,
    test_table: str,
) -> None:
    with pytest.raises(DatabaseTransactionError):
        with app_db.transaction() as conn:
            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 100)"))

            # Duplicate primary key must fail.
            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 200)"))

    assert count_records(app_db, test_table) == 0


def test_read_only_enforcement(
    app_db: Database,
    test_table: str,
) -> None:
    with pytest.raises(DatabaseTransactionError):
        with app_db.read_transaction() as conn:
            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 100)"))

    assert count_records(app_db, test_table) == 0


def test_application_cannot_create_tables(
    app_db: Database,
    migrator_db: Database,
) -> None:
    table = f"taqamind.permission_{uuid4().hex}"

    try:
        with pytest.raises(DatabaseTransactionError):
            with app_db.transaction() as conn:
                conn.execute(text(f"CREATE TABLE {table} (id INTEGER)"))
    finally:
        with migrator_db.transaction() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def test_migrator_can_create_tables(
    migrator_db: Database,
    test_table: str,
) -> None:
    assert count_records(migrator_db, test_table) == 0


def test_connection_returned_to_pool(
    app_db: Database,
) -> None:
    before = app_db.engine.pool.checkedout()

    with app_db.read_transaction() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1

        assert app_db.engine.pool.checkedout() == before + 1

    assert app_db.engine.pool.checkedout() == before


def test_no_dirty_reads(
    app_db: Database,
    test_table: str,
) -> None:
    with app_db.transaction() as writer:
        writer.execute(text(f"INSERT INTO {test_table} VALUES (1, 100)"))

        with app_db.read_transaction() as reader:
            count = reader.execute(text(f"SELECT COUNT(*) FROM {test_table}")).scalar_one()

            assert count == 0

    assert count_records(app_db, test_table) == 1


def test_concurrent_writes(
    app_db: Database,
    test_table: str,
) -> None:
    def insert_record(record_id: int) -> None:
        with app_db.transaction() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {test_table}
                    VALUES (:id, :value)
                    """
                ),
                {
                    "id": record_id,
                    "value": record_id * 10,
                },
            )

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(insert_record, range(50)))

    assert count_records(app_db, test_table) == 50


def test_recovery_after_failed_transaction(
    app_db: Database,
    test_table: str,
) -> None:
    with pytest.raises(DatabaseTransactionError):
        with app_db.transaction() as conn:
            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 100)"))

            conn.execute(text(f"INSERT INTO {test_table} VALUES (1, 200)"))

    with app_db.transaction() as conn:
        conn.execute(text(f"INSERT INTO {test_table} VALUES (2, 300)"))

    assert count_records(app_db, test_table) == 1


def test_transaction_modes(app_db: Database) -> None:
    with app_db.read_transaction() as conn:
        mode = conn.execute(text("SHOW transaction_read_only")).scalar_one()

        assert mode == "on"

    with app_db.transaction() as conn:
        mode = conn.execute(text("SHOW transaction_read_only")).scalar_one()

        assert mode == "off"
