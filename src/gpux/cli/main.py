"""Main CLI entry point for GPUX."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from gpux.cli.build import build_command
from gpux.cli.run import run_command
from gpux.cli.serve import serve_command
from gpux.cli.inspect import inspect_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False)]
)

# Create CLI app
app = typer.Typer(
    name="gpux",
    help="Docker-like GPU runtime for ML inference with universal GPU compatibility",
    add_completion=False,
    rich_markup_mode="rich",
)

# Add commands
app.command("build")(build_command)
app.command("run")(run_command)
app.command("serve")(serve_command)
app.command("inspect")(inspect_command)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
) -> None:
    """GPUX - Docker-like GPU runtime for ML inference.
    
    GPUX provides universal GPU compatibility for ML inference workloads,
    allowing you to run the same model on any GPU without compatibility issues.
    
    Examples:
        gpux build .                    # Build model from current directory
        gpux run sentiment-analysis     # Run inference on a model
        gpux serve model-name --port 8080  # Start HTTP server
        gpux inspect model-name        # Inspect model information
    """
    if version:
        from gpux import __version__
        typer.echo(f"GPUX version {__version__}")
        raise typer.Exit()
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\n[red]Interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        typer.echo(f"[red]Error: {e}[/red]")
        sys.exit(1)
