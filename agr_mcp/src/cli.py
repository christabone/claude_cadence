"""Command-line interface for AGR MCP Server.

This module provides CLI commands for running and managing the AGR MCP server.
"""

import asyncio
import click
from pathlib import Path
from typing import Optional

from .config import get_config
from .server import AGRMCPServer
from .utils.logging_config import setup_logging


@click.group()
@click.version_option()
def cli():
    """AGR MCP Server - Alliance Genome Resources data access via MCP."""
    pass


@cli.command()
@click.option(
    "--host",
    default="localhost",
    help="Host to bind the server to"
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to listen on"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def serve(host: str, port: int, config: Optional[str], debug: bool):
    """Run the AGR MCP server."""
    # Load configuration
    config_obj = get_config(config)

    # Override with CLI options
    if debug:
        config_obj.server.debug = True
        config_obj.logging.level = "DEBUG"

    config_obj.server.host = host
    config_obj.server.port = port

    # Setup logging
    setup_logging(config_obj.logging.dict())

    # Create and run server
    server = AGRMCPServer(config_obj.dict())

    click.echo(f"Starting AGR MCP Server on {host}:{port}")
    if debug:
        click.echo("Debug mode enabled")

    try:
        asyncio.run(server.run(host, port))
    except KeyboardInterrupt:
        click.echo("\nShutting down server...")


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="config.yml",
    help="Output path for configuration file"
)
def init(output: str):
    """Initialize a new configuration file."""
    import yaml

    default_config = {
        "server": {
            "host": "localhost",
            "port": 8000,
            "debug": False
        },
        "api": {
            "base_url": "https://www.alliancegenome.org/api",
            "timeout": 30,
            "max_retries": 3,
            "rate_limit": 10
        },
        "logging": {
            "level": "INFO",
            "file": "logs/agr_mcp.log",
            "console": True,
            "rich": True
        },
        "download_dir": str(Path.home() / ".agr_mcp" / "downloads"),
        "cache_dir": str(Path.home() / ".agr_mcp" / "cache")
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Configuration file created at: {output_path}")


@cli.command()
def validate():
    """Validate server configuration and connectivity."""
    import asyncio
    from .tools.api_schema import get_api_status

    click.echo("Validating configuration...")

    try:
        # Load configuration
        config = get_config()
        click.echo("✓ Configuration loaded successfully")

        # Setup logging
        setup_logging(config.logging.dict())
        click.echo("✓ Logging configured")

        # Test API connectivity
        click.echo("\nTesting AGR API connectivity...")

        async def check_status():
            return await get_api_status()

        status = asyncio.run(check_status())

        if status["status"] == "online":
            click.echo(f"✓ API is online (response time: {status.get('response_time_ms', 'N/A')}ms)")
        else:
            click.echo(f"✗ API is {status['status']}")
            if "error" in status:
                click.echo(f"  Error: {status['error']}")

        click.echo("\nValidation complete!")

    except Exception as e:
        click.echo(f"✗ Validation failed: {str(e)}", err=True)
        raise click.Abort()


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
