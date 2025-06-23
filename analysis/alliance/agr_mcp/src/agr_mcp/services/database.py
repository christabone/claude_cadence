"""
Database service for AGR PostgreSQL data access.

This module provides functionality for connecting to and querying
Alliance PostgreSQL databases (both AWS and local instances).
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator

import asyncpg
from asyncpg.pool import Pool

from ..core.config import DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a database query."""
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time: float


class DatabaseService:
    """
    Service for PostgreSQL database operations.

    Provides connection pooling, query execution, and result formatting
    for Alliance database access.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database service.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[Pool] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if self._pool is not None:
            return

        try:
            dsn = self._build_dsn()
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=10,
                command_timeout=60,
                server_settings={
                    'application_name': 'agr-mcp-server'
                }
            )
            logger.info(f"Connected to database: {self.config.name}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from database")

    def _build_dsn(self) -> str:
        """Build PostgreSQL DSN from configuration."""
        return (
            f"postgresql://{self.config.user}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.name}"
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a database connection from the pool.

        Yields:
            Database connection
        """
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as conn:
            yield conn

    async def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        timeout: Optional[float] = None
    ) -> QueryResult:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters
            timeout: Query timeout in seconds

        Returns:
            QueryResult with columns, rows, and metadata
        """
        import time
        start_time = time.time()

        async with self.acquire() as conn:
            try:
                # Set query timeout if specified
                if timeout:
                    await conn.execute(f"SET statement_timeout = {int(timeout * 1000)}")

                # Execute query
                result = await conn.fetch(query, *(params or []))

                # Convert to list of dicts
                rows = [dict(record) for record in result]
                columns = list(result[0].keys()) if result else []

                execution_time = time.time() - start_time

                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time=execution_time
                )

            except asyncpg.QueryCanceledError:
                raise TimeoutError(f"Query exceeded timeout of {timeout}s")
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise

    async def list_tables(self, schema: str = "public") -> List[str]:
        """
        List all tables in the specified schema.

        Args:
            schema: Database schema name

        Returns:
            List of table names
        """
        query = """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = $1
            ORDER BY tablename
        """

        result = await self.execute_query(query, [schema])
        return [row["tablename"] for row in result.rows]

    async def describe_table(self, table_name: str, schema: str = "public") -> Dict[str, Any]:
        """
        Get detailed information about a table.

        Args:
            table_name: Name of the table
            schema: Database schema name

        Returns:
            Dictionary with table metadata
        """
        # Get column information
        columns_query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """

        columns_result = await self.execute_query(columns_query, [schema, table_name])

        # Get primary key information
        pk_query = """
            SELECT a.attname as column_name
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = ($1||'.'||$2)::regclass AND i.indisprimary
        """

        pk_result = await self.execute_query(pk_query, [schema, table_name])
        pk_columns = [row["column_name"] for row in pk_result.rows]

        # Get foreign key information
        fk_query = """
            SELECT
                kcu.column_name,
                ccu.table_schema AS foreign_table_schema,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2
        """

        fk_result = await self.execute_query(fk_query, [schema, table_name])

        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {schema}.{table_name}"
        count_result = await self.execute_query(count_query)
        row_count = count_result.rows[0]["count"] if count_result.rows else 0

        return {
            "schema": schema,
            "table_name": table_name,
            "columns": columns_result.rows,
            "primary_key": pk_columns,
            "foreign_keys": fk_result.rows,
            "row_count": row_count
        }

    async def export_query_results(
        self,
        query: str,
        format: str = "json",
        params: Optional[List[Any]] = None
    ) -> str:
        """
        Export query results in various formats.

        Args:
            query: SQL query to execute
            format: Export format (json, csv, tsv)
            params: Query parameters

        Returns:
            Formatted query results
        """
        result = await self.execute_query(query, params)

        if format == "json":
            import json
            return json.dumps(result.rows, indent=2, default=str)

        elif format in ("csv", "tsv"):
            import csv
            import io

            output = io.StringIO()
            delimiter = "\t" if format == "tsv" else ","

            writer = csv.DictWriter(
                output,
                fieldnames=result.columns,
                delimiter=delimiter
            )
            writer.writeheader()
            writer.writerows(result.rows)

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported export format: {format}")
