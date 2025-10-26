"""Serve command for GPUX CLI."""

from __future__ import annotations

import logging
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from gpux.config.parser import GPUXConfigParser
from gpux.core.discovery import ModelDiscovery
from gpux.core.managers.exceptions import ModelNotFoundError
from gpux.core.runtime import GPUXRuntime

console = Console()
logger = logging.getLogger(__name__)

serve_app = typer.Typer(name="serve", help="Start HTTP server for model serving")


@serve_app.command()
def serve_command(
    model_name: str = typer.Argument(
        ...,
        help="Name of the model to serve",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port to serve on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to serve on",
    ),
    config_file: str = typer.Option(
        "gpux.yml",
        "--config",
        "-c",
        help="Configuration file name",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Preferred execution provider",
    ),
    workers: int = typer.Option(
        default=1,
        help="Number of worker processes",
    ),
    *,
    verbose: bool = typer.Option(
        default=False,
        help="Enable verbose output",
    ),
) -> None:
    """Start HTTP server for model serving.

    This command starts a FastAPI server that provides REST API endpoints
    for model inference.

    Examples:
        gpux serve sentiment-analysis
        gpux serve image-classifier --port 9000
        gpux serve model-name --host 127.0.0.1 --workers 4
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Find model configuration using unified discovery
        model_path = ModelDiscovery.find_model_config(model_name, config_file)

        # Parse configuration
        parser = GPUXConfigParser()
        config = parser.parse_file(model_path / config_file)

        # Get model file path
        model_file = parser.get_model_path(model_path)
        if not model_file or not model_file.exists():
            console.print(f"[red]Error: Model file not found: {model_file}[/red]")
            raise typer.Exit(1) from None

        # Initialize runtime
        console.print("[blue]Loading model...[/blue]")
        runtime = GPUXRuntime(
            model_path=model_file,
            provider=provider,
            **config.runtime.dict(),
        )

        # Display server information
        _display_server_info(config, model_name, host, port, workers)

        # Start server
        _start_server(runtime, config, host, port, workers)

    except ModelNotFoundError as e:
        console.print(f"[red]{e.format_error_message()}[/red]")
        raise typer.Exit(1) from e
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        console.print(f"[red]Serve failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


def _display_server_info(
    config: Any, model_name: str, host: str, port: int, workers: int
) -> None:
    """Display server information."""

    # Model information
    model_table = Table(
        title="Model Information",
        show_header=True,
        header_style="bold magenta",
    )
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="white")

    model_table.add_row("Name", model_name)
    model_table.add_row("Version", config.version)
    model_table.add_row("Inputs", str(len(config.inputs)))
    model_table.add_row("Outputs", str(len(config.outputs)))

    console.print(model_table)

    # Server information
    server_table = Table(
        title="Server Configuration",
        show_header=True,
        header_style="bold green",
    )
    server_table.add_column("Property", style="cyan")
    server_table.add_column("Value", style="white")

    server_table.add_row("Host", host)
    server_table.add_row("Port", str(port))
    server_table.add_row("Workers", str(workers))
    server_table.add_row("URL", f"http://{host}:{port}")

    console.print(server_table)

    # API endpoints
    endpoints_table = Table(
        title="API Endpoints",
        show_header=True,
        header_style="bold blue",
    )
    endpoints_table.add_column("Method", style="cyan")
    endpoints_table.add_column("Path", style="white")
    endpoints_table.add_column("Description", style="white")

    endpoints_table.add_row("POST", "/predict", "Run inference")
    endpoints_table.add_row("GET", "/health", "Health check")
    endpoints_table.add_row("GET", "/info", "Model information")
    endpoints_table.add_row("GET", "/metrics", "Performance metrics")

    console.print(endpoints_table)


def _start_server(  # noqa: C901
    runtime: GPUXRuntime, config: Any, host: str, port: int, workers: int
) -> None:
    """Start the HTTP server.

    Args:
        runtime: GPUX runtime instance
        config: Model configuration
        host: Host to serve on
        port: Port to serve on
        workers: Number of workers
    """
    try:
        import numpy as np
        import uvicorn
        from fastapi import (
            FastAPI,
            HTTPException,
        )

        # Create FastAPI app
        app = FastAPI(
            title=f"GPUX Server - {config.name}",
            description=f"ML inference server for {config.name}",
            version=config.version,
        )

        # Health check endpoint
        @app.get("/health")
        async def health_check() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy", "model": config.name}

        # Model info endpoint
        @app.get("/info")
        async def model_info() -> dict[str, Any]:
            """Get model information."""
            model_info = runtime.get_model_info()
            if not model_info:
                raise HTTPException(status_code=500, detail="Model not loaded")

            return model_info.to_dict()

        # Metrics endpoint
        @app.get("/metrics")
        async def metrics() -> dict[str, Any]:
            """Get performance metrics."""
            provider_info = runtime.get_provider_info()
            return {
                "provider": provider_info,
                "available_providers": runtime.get_available_providers(),
            }

        # Prediction endpoint
        @app.post("/predict")
        async def predict(data: dict[str, Any]) -> dict[str, Any]:
            """Run inference on input data."""
            try:
                # Convert input data to numpy arrays
                numpy_input = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        numpy_input[key] = np.array(value)
                    else:
                        numpy_input[key] = value

                # Run inference
                results = runtime.infer(numpy_input)
            except (ValueError, RuntimeError, KeyError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            else:
                # Convert results to JSON-serializable format
                output_data = {}
                for key, value in results.items():
                    if hasattr(value, "tolist"):
                        output_data[key] = value.tolist()
                    else:
                        output_data[key] = value

                return output_data

        # Start server
        console.print("\n[green]🚀 Starting GPUX server...[/green]")
        console.print(f"[dim]Server will be available at: http://{host}:{port}[/dim]")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")

        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=workers if workers > 1 else None,
            log_level=(
                "info"
                if not logging.getLogger().isEnabledFor(logging.DEBUG)
                else "debug"
            ),
        )

    except ImportError as e:
        console.print("[red]Error: FastAPI and uvicorn are required for serving[/red]")
        console.print("[yellow]Install with: pip install fastapi uvicorn[/yellow]")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    finally:
        runtime.cleanup()
