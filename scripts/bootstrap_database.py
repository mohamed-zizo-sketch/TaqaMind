"""Initialize and verify local TaqaMind PostgreSQL infrastructure."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ADMIN_USER = "taqamind_admin"
SCHEMA_NAME = "taqamind"

BOOTSTRAP_LOCK_ID = 714082611

INVALID_PASSWORDS = {
    "",
    "CHANGE_ME",
    "YOUR_APP_PASSWORD",
    "YOUR_MIGRATOR_PASSWORD",
    "YOUR_TEST_APP_PASSWORD",
    "YOUR_TEST_MIGRATOR_PASSWORD",
}


class BootstrapError(Exception):
    """Raised when database infrastructure cannot be initialized safely."""


@dataclass(frozen=True, slots=True)
class RoleSettings:
    username: str
    password: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class EnvironmentSettings:
    host: str
    port: int
    database: str
    application: RoleSettings
    migrator: RoleSettings


@dataclass(frozen=True, slots=True)
class AdminSettings:
    host: str
    port: int
    username: str
    password: str = field(repr=False)


def read_environment(path: Path) -> dict[str, str | None]:
    """Read an explicit environment file without modifying os.environ."""

    if not path.is_file():
        raise BootstrapError(f"Missing configuration file: {path.name}")

    return dict(
        dotenv_values(
            dotenv_path=path,
            interpolate=False,
        )
    )


def require_value(
    settings: dict[str, str | None],
    key: str,
) -> str:
    value = settings.get(key)

    if value is None or not value.strip():
        raise BootstrapError(f"Missing configuration value: {key}")

    return value


def require_password(
    settings: dict[str, str | None],
    key: str,
) -> str:
    value = require_value(settings, key)

    if value.strip() in INVALID_PASSWORDS:
        raise BootstrapError(f"Password must be configured: {key}")

    return value


def load_environment(
    settings: dict[str, str | None],
    *,
    expected_database: str,
    expected_app: str,
    expected_migrator: str,
) -> EnvironmentSettings:
    """Validate the expected local database configuration."""

    host = require_value(settings, "POSTGRES_HOST")

    if host != "127.0.0.1":
        raise BootstrapError("Bootstrap supports local PostgreSQL only.")

    try:
        port = int(require_value(settings, "POSTGRES_PORT"))
    except ValueError:
        raise BootstrapError("Invalid PostgreSQL port.") from None

    if not 1 <= port <= 65535:
        raise BootstrapError("PostgreSQL port is out of range.")

    database = require_value(settings, "POSTGRES_DB")
    app_user = require_value(settings, "DB_APP_USER")
    migrator_user = require_value(settings, "DB_MIGRATOR_USER")

    if database != expected_database:
        raise BootstrapError("Unexpected database name.")

    if app_user != expected_app:
        raise BootstrapError("Unexpected application role.")

    if migrator_user != expected_migrator:
        raise BootstrapError("Unexpected migration role.")

    if settings.get("DB_SCHEMA", SCHEMA_NAME) != SCHEMA_NAME:
        raise BootstrapError("Unexpected database schema.")

    return EnvironmentSettings(
        host=host,
        port=port,
        database=database,
        application=RoleSettings(
            username=app_user,
            password=require_password(settings, "DB_APP_PASSWORD"),
        ),
        migrator=RoleSettings(
            username=migrator_user,
            password=require_password(settings, "DB_MIGRATOR_PASSWORD"),
        ),
    )


def load_settings():
    """Load and validate both environments before making changes."""

    main_values = read_environment(PROJECT_ROOT / ".env")
    test_values = read_environment(PROJECT_ROOT / ".env.test")

    main = load_environment(
        main_values,
        expected_database="taqamind",
        expected_app="taqamind_app",
        expected_migrator="taqamind_migrator",
    )

    test = load_environment(
        test_values,
        expected_database="taqamind_test",
        expected_app="taqamind_test_app",
        expected_migrator="taqamind_test_migrator",
    )

    if (main.host, main.port) != (test.host, test.port):
        raise BootstrapError("Main and test databases must use the same local server.")

    admin_user = require_value(main_values, "POSTGRES_USER")

    if admin_user != ADMIN_USER:
        raise BootstrapError("Unexpected administrative role.")

    admin = AdminSettings(
        host=main.host,
        port=main.port,
        username=admin_user,
        password=require_password(main_values, "POSTGRES_PASSWORD"),
    )

    return admin, main, test


def connect_as_admin(
    admin: AdminSettings,
    database: str = "postgres",
) -> psycopg.Connection:
    """Open an administrative connection for infrastructure operations."""

    return psycopg.connect(
        host=admin.host,
        port=admin.port,
        dbname=database,
        user=admin.username,
        password=admin.password,
        connect_timeout=10,
        autocommit=True,
    )


def verify_admin(conn: psycopg.Connection) -> None:
    row = conn.execute(
        """
        SELECT current_user, rolsuper
        FROM pg_roles
        WHERE rolname = current_user
        """
    ).fetchone()

    if row != (ADMIN_USER, True):
        raise BootstrapError("Expected administrative role is required.")


def ensure_role(
    conn: psycopg.Connection,
    role: RoleSettings,
) -> None:
    """Create a missing role or validate an existing role."""

    row = conn.execute(
        """
        SELECT
            rolcanlogin,
            rolsuper,
            rolcreatedb,
            rolcreaterole,
            rolreplication,
            rolbypassrls
        FROM pg_roles
        WHERE rolname = %s
        """,
        (role.username,),
    ).fetchone()

    if row is None:
        conn.execute("SET password_encryption = 'scram-sha-256'")

        conn.execute(
            sql.SQL(
                """
                CREATE ROLE {}
                WITH LOGIN
                NOSUPERUSER
                NOCREATEDB
                NOCREATEROLE
                NOREPLICATION
                NOBYPASSRLS
                PASSWORD {}
                """
            ).format(
                sql.Identifier(role.username),
                sql.Literal(role.password),
            )
        )

        print(f"[CREATED] Role: {role.username}")

    else:
        expected = (True, False, False, False, False, False)

        if row != expected:
            raise BootstrapError(f"Unsafe role attributes: {role.username}")

        print(f"[EXISTS] Role: {role.username}")

    memberships = conn.execute(
        """
        SELECT COUNT(*)
        FROM pg_auth_members AS membership
        JOIN pg_roles AS member
          ON membership.member = member.oid
        WHERE member.rolname = %s
        """,
        (role.username,),
    ).fetchone()[0]

    if memberships:
        raise BootstrapError(f"Unexpected role memberships: {role.username}")


def ensure_database(
    conn: psycopg.Connection,
    database: str,
) -> None:
    """Create a missing database without dropping existing data."""

    row = conn.execute(
        """
        SELECT pg_get_userbyid(datdba)
        FROM pg_database
        WHERE datname = %s
        """,
        (database,),
    ).fetchone()

    if row is None:
        conn.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(database),
                sql.Identifier(ADMIN_USER),
            )
        )

        print(f"[CREATED] Database: {database}")

    else:
        if row[0] != ADMIN_USER:
            raise BootstrapError(f"Unexpected database owner: {database}")

        print(f"[EXISTS] Database: {database}")


def configure_database_access(
    conn: psycopg.Connection,
    environment: EnvironmentSettings,
) -> None:
    """Apply the expected database-level access policy."""

    conn.execute(
        sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(
            sql.Identifier(environment.database)
        )
    )

    for role in (environment.application, environment.migrator):
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(environment.database),
                sql.Identifier(role.username),
            )
        )

    print(f"[CONFIGURED] Database access: {environment.database}")


def configure_application_timeouts(
    conn: psycopg.Connection,
    environment: EnvironmentSettings,
) -> None:
    """Configure safe transaction timeouts for application roles."""

    timeout_settings = {
        "statement_timeout": "60s",
        "lock_timeout": "10s",
        "idle_in_transaction_session_timeout": "60s",
    }

    for setting, value in timeout_settings.items():
        conn.execute(
            sql.SQL("ALTER ROLE {} IN DATABASE {} SET {} = {}").format(
                sql.Identifier(environment.application.username),
                sql.Identifier(environment.database),
                sql.Identifier(setting),
                sql.Literal(value),
            )
        )

    print(f"[CONFIGURED] Application timeouts: {environment.application.username}")


def configure_schema(
    admin: AdminSettings,
    environment: EnvironmentSettings,
) -> None:
    """Initialize the schema and its least-privilege permissions."""

    with connect_as_admin(admin, environment.database) as conn:
        verify_admin(conn)

        with conn.transaction():
            row = conn.execute(
                """
                SELECT pg_get_userbyid(nspowner)
                FROM pg_namespace
                WHERE nspname = %s
                """,
                (SCHEMA_NAME,),
            ).fetchone()

            if row is None:
                conn.execute(
                    sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                        sql.Identifier(SCHEMA_NAME),
                        sql.Identifier(environment.migrator.username),
                    )
                )

                print(f"[CREATED] Schema: {environment.database}.{SCHEMA_NAME}")

            elif row[0] != environment.migrator.username:
                raise BootstrapError(f"Unexpected schema owner: {environment.database}")

            conn.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")

            conn.execute(
                sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(SCHEMA_NAME))
            )

            conn.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    sql.Identifier(SCHEMA_NAME),
                    sql.Identifier(environment.application.username),
                )
            )

            for role in (
                environment.application,
                environment.migrator,
            ):
                conn.execute(
                    sql.SQL(
                        """
                        ALTER ROLE {}
                        IN DATABASE {}
                        SET search_path TO {}
                        """
                    ).format(
                        sql.Identifier(role.username),
                        sql.Identifier(environment.database),
                        sql.Identifier(SCHEMA_NAME),
                    )
                )

            privileges = conn.execute(
                """
                SELECT
                    has_schema_privilege(%s, %s, 'USAGE'),
                    has_schema_privilege(%s, %s, 'CREATE')
                """,
                (
                    environment.application.username,
                    SCHEMA_NAME,
                    environment.application.username,
                    SCHEMA_NAME,
                ),
            ).fetchone()

            if privileges != (True, False):
                raise BootstrapError("Unsafe application schema permissions.")

    print(f"[CONFIGURED] Schema: {environment.database}.{SCHEMA_NAME}")


def verify_database_isolation(
    conn: psycopg.Connection,
    main: EnvironmentSettings,
    test: EnvironmentSettings,
) -> None:
    """Verify that application and test accounts cannot cross databases."""

    for environment in (main, test):
        for role in (environment.application, environment.migrator):
            row = conn.execute(
                """
                SELECT
                    has_database_privilege(%s, %s, 'CONNECT'),
                    has_database_privilege(%s, %s, 'CONNECT')
                """,
                (
                    role.username,
                    main.database,
                    role.username,
                    test.database,
                ),
            ).fetchone()

            expected = (True, False) if environment.database == main.database else (False, True)

            if row != expected:
                raise BootstrapError(f"Database isolation failed: {role.username}")

            print(f"[PASSED] Database isolation: {role.username}")


def verify_authentication(
    environment: EnvironmentSettings,
) -> None:
    """Verify actual connections using application and migration roles."""

    for role in (environment.application, environment.migrator):
        try:
            with psycopg.connect(
                host=environment.host,
                port=environment.port,
                dbname=environment.database,
                user=role.username,
                password=role.password,
                connect_timeout=10,
            ) as conn:
                row = conn.execute(
                    """
                    SELECT
                        current_database(),
                        current_user,
                        current_schema()
                    """
                ).fetchone()

                expected = (
                    environment.database,
                    role.username,
                    SCHEMA_NAME,
                )

                if row != expected:
                    raise BootstrapError(f"Unexpected connection identity: {role.username}")

        except psycopg.Error:
            raise BootstrapError(f"Authentication or connection failed: {role.username}") from None

        print(f"[PASSED] Connection: {role.username}")


def verify_application_timeouts(
    environment: EnvironmentSettings,
) -> None:
    """Verify effective application transaction timeouts."""

    role = environment.application

    with psycopg.connect(
        host=environment.host,
        port=environment.port,
        dbname=environment.database,
        user=role.username,
        password=role.password,
        connect_timeout=10,
    ) as conn:
        row = conn.execute(
            """
            SELECT
                current_setting('statement_timeout'),
                current_setting('lock_timeout'),
                current_setting(
                    'idle_in_transaction_session_timeout'
                )
            """
        ).fetchone()

        if row != ("1min", "10s", "1min"):
            raise BootstrapError(f"Unexpected application timeouts: {role.username}")

    print(f"[PASSED] Application timeouts: {role.username}")


def bootstrap() -> None:
    """Initialize and verify the local database infrastructure."""

    admin, main, test = load_settings()

    print("[PASSED] Environment configuration")

    with connect_as_admin(admin) as conn:
        verify_admin(conn)

        acquired = conn.execute(
            "SELECT pg_try_advisory_lock(%s)",
            (BOOTSTRAP_LOCK_ID,),
        ).fetchone()[0]

        if not acquired:
            raise BootstrapError("Another database bootstrap is already running.")

        try:
            for environment in (main, test):
                ensure_role(conn, environment.application)
                ensure_role(conn, environment.migrator)

            for environment in (main, test):
                ensure_database(conn, environment.database)

            for environment in (main, test):
                configure_database_access(conn, environment)
                configure_application_timeouts(conn, environment)

            for environment in (main, test):
                configure_schema(admin, environment)

            verify_database_isolation(conn, main, test)

            verify_authentication(main)
            verify_authentication(test)
            verify_application_timeouts(main)
            verify_application_timeouts(test)

        finally:
            conn.execute(
                "SELECT pg_advisory_unlock(%s)",
                (BOOTSTRAP_LOCK_ID,),
            )

    print("\nDatabase bootstrap completed successfully.")


def main() -> int:
    try:
        bootstrap()

    except BootstrapError as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 1

    except psycopg.Error as exc:
        print(
            f"[FAILED] PostgreSQL operation failed. SQLSTATE: {exc.sqlstate or 'UNAVAILABLE'}",
            file=sys.stderr,
        )
        return 1

    except (OSError, UnicodeError):
        print(
            "[FAILED] Unable to read configuration files.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
