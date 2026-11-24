"""Idempotent Neon Postgres + pgvector schema migration and verification runner.

Usage:
    python -m backend.app.db.migrate

What it does:
    1. Reads `backend/app/db/schema.sql`.
    2. Executes each DDL statement idempotently inside a single async transaction.
    3. Verifies that `pgvector`, all 3 tables (`users`, `documents`, `chunks`),
       the default system user, and all 5 required indexes exist in PostgreSQL.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text

from backend.app.db.session import check_db_health, dispose_engine, get_engine

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

EXPECTED_TABLES = {"users", "documents", "chunks"}
EXPECTED_INDEXES = {
    "idx_chunks_embedding_hnsw",
    "idx_chunks_tsv_gin",
    "idx_chunks_document_id",
    "idx_chunks_user_id",
    "idx_documents_user_id",
}


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split a SQL script into executable semicolon-terminated statements.

    Strips full-line `--` comments before splitting so `asyncpg` (which executes
    one prepared statement per call) receives clean SQL commands.

    Args:
        sql_text: Raw SQL file content.

    Returns:
        Ordered list of non-empty SQL statements.
    """
    cleaned_lines = [
        line for line in sql_text.splitlines() if not line.strip().startswith("--")
    ]
    cleaned_sql = "\n".join(cleaned_lines)
    return [stmt.strip() for stmt in cleaned_sql.split(";") if stmt.strip()]


async def run_migration() -> dict[str, object]:
    """Apply `schema.sql` and verify all tables, indexes, and extensions.

    Returns:
        Verification summary dictionary detailing `pgvector_version`,
        `tables`, `indexes`, and `default_user_seeded`.

    Raises:
        FileNotFoundError: If `schema.sql` is missing.
        RuntimeError: If any expected table, index, or extension is missing
            after applying the migration.
    """
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    statements = _split_sql_statements(SCHEMA_PATH.read_text(encoding="utf-8"))
    engine = get_engine()

    # Execute all DDL statements atomically inside one transaction.
    async with engine.begin() as conn:
        for stmt in statements:
            await conn.execute(text(stmt))

    # Run post-migration verification queries against PostgreSQL catalogs.
    health = await check_db_health(timeout_seconds=10.0)

    async with engine.connect() as conn:
        tables_res = await conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename IN ('users', 'documents', 'chunks')"
            )
        )
        found_tables = {row[0] for row in tables_res.fetchall()}

        idx_res = await conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'public' AND tablename IN ('documents', 'chunks')"
            )
        )
        found_indexes = {row[0] for row in idx_res.fetchall()}

        user_res = await conn.execute(
            text(
                "SELECT id FROM users "
                "WHERE id = '00000000-0000-0000-0000-000000000000'::uuid"
            )
        )
        default_user_seeded = user_res.scalar_one_or_none() is not None

    missing_tables = EXPECTED_TABLES - found_tables
    missing_indexes = EXPECTED_INDEXES - found_indexes
    if missing_tables or missing_indexes or not default_user_seeded:
        raise RuntimeError(
            "Schema verification failed: "
            f"missing_tables={sorted(missing_tables)}, "
            f"missing_indexes={sorted(missing_indexes)}, "
            f"default_user_seeded={default_user_seeded}"
        )

    return {
        "pgvector_version": health["pgvector_version"],
        "tables": sorted(found_tables),
        "indexes": sorted(found_indexes & EXPECTED_INDEXES),
        "default_user_seeded": default_user_seeded,
    }


async def _main() -> None:
    """CLI entry point for running migrations and disposing the async pool."""
    try:
        summary = await run_migration()
        print("Migration & verification succeeded:")
        print(f"  pgvector version : {summary['pgvector_version']}")
        print(f"  tables verified  : {', '.join(summary['tables'])}")
        print(f"  indexes verified : {', '.join(summary['indexes'])}")
        print(f"  default user     : {summary['default_user_seeded']}")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(_main())
