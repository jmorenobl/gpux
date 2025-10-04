"""Inspect command for GPUX CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from gpux.config.parser import GPUXConfigParser
from gpux.core.models import ModelInspector
from gpux.core.providers import ProviderManager

console = Console()
logger = logging.getLogger(__name__)

inspect_app = typer.Typer(name="inspect", help="Inspect models and runtime information")


@inspect_app.command()
def inspect_command(
    model_name: str | None = typer.Argument(
        None,
        help="Name of the model to inspect",
    ),
    config_file: str = typer.Option(
        "gpux.yml",
        "--config",
        "-c",
        help="Configuration file name",
    ),
    model_file: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Direct path to model file",
    ),
    *,
    json_output: bool = typer.Option(
        default=False,
        help="Output in JSON format",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="Enable verbose output",
    ),
) -> None:
    """Inspect models and runtime information.

    This command provides detailed information about models, their inputs/outputs,
    and available execution providers.

    Examples:
        gpux inspect sentiment-analysis
        gpux inspect --model ./model.onnx
        gpux inspect --json
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if model_file:
            # Inspect model file directly
            _inspect_model_file(Path(model_file), json_output=json_output)
        elif model_name:
            # Inspect model by name
            _inspect_model_by_name(model_name, config_file, json_output=json_output)
        else:
            # Show runtime information
            _inspect_runtime(json_output=json_output)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Inspect failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from e


def _inspect_model_file(model_path: Path, *, json_output: bool) -> None:
    """Inspect a model file directly.

    Args:
        model_path: Path to the model file
        json_output: Whether to output in JSON format
    """
    if not model_path.exists():
        console.print(f"[red]Error: Model file not found: {model_path}[/red]")
        raise typer.Exit(1) from None

    # Inspect model
    inspector = ModelInspector()
    model_info = inspector.inspect(model_path)

    if json_output:
        console.print(JSON.from_data(model_info.to_dict()))
    else:
        _display_model_info(model_info)


def _inspect_model_by_name(
    model_name: str, config_file: str, *, json_output: bool
) -> None:
    """Inspect a model by name.

    Args:
        model_name: Name of the model
        config_file: Configuration file name
        json_output: Whether to output in JSON format
    """
    # Find model configuration
    model_path = _find_model_config(model_name, config_file)
    if not model_path:
        console.print(f"[red]Error: Model '{model_name}' not found[/red]")
        raise typer.Exit(1) from None

    # Parse configuration
    parser = GPUXConfigParser()
    config = parser.parse_file(model_path / config_file)

    # Get model file path
    model_file = parser.get_model_path(model_path)
    if not model_file or not model_file.exists():
        console.print(f"[red]Error: Model file not found: {model_file}[/red]")
        raise typer.Exit(1) from None

    # Inspect model
    inspector = ModelInspector()
    model_info = inspector.inspect(model_file)

    if json_output:
        output_data = {
            "config": config.dict(),
            "model_info": model_info.to_dict(),
        }
        console.print(JSON.from_data(output_data))
    else:
        _display_config_info(config)
        _display_model_info(model_info)


def _inspect_runtime(*, json_output: bool) -> None:
    """Inspect runtime information.

    Args:
        json_output: Whether to output in JSON format
    """
    provider_manager = ProviderManager()
    available_providers = provider_manager.get_available_providers()

    if json_output:
        runtime_info: dict[str, Any] = {
            "available_providers": available_providers,
            "provider_details": {},
        }

        for provider in provider_manager._provider_priority:  # noqa: SLF001  # noqa: SLF001
            info = provider_manager.get_provider_info(provider)
            runtime_info["provider_details"][provider.value] = info

        console.print(JSON.from_data(runtime_info))
    else:
        _display_runtime_info(provider_manager)


def _find_model_config(model_name: str, config_file: str) -> Path | None:
    """Find model configuration file.

    Args:
        model_name: Name of the model
        config_file: Configuration file name

    Returns:
        Path to model directory or None if not found
    """
    # Check current directory
    current_dir = Path()
    if (current_dir / config_file).exists():
        return current_dir

    # Check if model_name is a directory
    model_dir = Path(model_name)
    if model_dir.is_dir() and (model_dir / config_file).exists():
        return model_dir

    # Check .gpux directory for built models
    gpux_dir = Path(".gpux")
    if gpux_dir.exists():
        # Look for model info files
        for info_file in gpux_dir.glob("**/model_info.json"):
            try:
                with info_file.open() as f:
                    info = json.load(f)
                if info.get("name") == model_name:
                    return info_file.parent.parent
            except (json.JSONDecodeError, OSError):
                continue

    return None


def _display_config_info(config: Any) -> None:
    """Display configuration information."""

    config_table = Table(
        title="Configuration",
        show_header=True,
        header_style="bold magenta",
    )
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Name", config.name)
    config_table.add_row("Version", config.version)
    config_table.add_row("Model Source", str(config.model.source))
    config_table.add_row("Model Format", config.model.format)
    config_table.add_row("GPU Memory", config.runtime.gpu.memory)
    config_table.add_row("GPU Backend", config.runtime.gpu.backend)
    config_table.add_row("Batch Size", str(config.runtime.batch_size))
    config_table.add_row("Timeout", f"{config.runtime.timeout}s")

    console.print(config_table)


def _display_model_info(model_info: Any) -> None:
    """Display model information."""

    # Basic model information
    model_table = Table(
        title="Model Information",
        show_header=True,
        header_style="bold magenta",
    )
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="white")

    model_table.add_row("Name", model_info.name)
    model_table.add_row("Version", model_info.version)
    model_table.add_row("Format", model_info.format)
    model_table.add_row("Size", f"{model_info.size_mb:.1f} MB")
    model_table.add_row("Path", str(model_info.path))

    console.print(model_table)

    # Input specifications
    if model_info.inputs:
        inputs_table = Table(
            title="Input Specifications",
            show_header=True,
            header_style="bold green",
        )
        inputs_table.add_column("Name", style="cyan")
        inputs_table.add_column("Type", style="white")
        inputs_table.add_column("Shape", style="white")
        inputs_table.add_column("Required", style="white")
        inputs_table.add_column("Description", style="white")

        for inp in model_info.inputs:
            inputs_table.add_row(
                inp.name,
                inp.type,
                str(inp.shape) if inp.shape else "Dynamic",
                "✅" if inp.required else "❌",
                inp.description or "N/A",
            )

        console.print(inputs_table)

    # Output specifications
    if model_info.outputs:
        outputs_table = Table(
            title="Output Specifications",
            show_header=True,
            header_style="bold green",
        )
        outputs_table.add_column("Name", style="cyan")
        outputs_table.add_column("Type", style="white")
        outputs_table.add_column("Shape", style="white")
        outputs_table.add_column("Labels", style="white")
        outputs_table.add_column("Description", style="white")

        for out in model_info.outputs:
            labels_str = str(out.labels) if out.labels else "N/A"
            outputs_table.add_row(
                out.name,
                out.type,
                str(out.shape) if out.shape else "Dynamic",
                labels_str,
                out.description or "N/A",
            )

        console.print(outputs_table)

    # Metadata
    if model_info.metadata:
        metadata_table = Table(
            title="Metadata",
            show_header=True,
            header_style="bold blue",
        )
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value", style="white")

        for key, value in model_info.metadata.items():
            value_str = str(value)
            metadata_table.add_row(key, value_str)

        console.print(metadata_table)


def _display_runtime_info(provider_manager: ProviderManager) -> None:
    """Display runtime information."""

    # Available providers
    providers_table = Table(
        title="Available Execution Providers",
        show_header=True,
        header_style="bold magenta",
    )
    providers_table.add_column("Provider", style="cyan")
    providers_table.add_column("Available", style="white")
    providers_table.add_column("Platform", style="white")
    providers_table.add_column("Description", style="white")

    for provider in provider_manager._provider_priority:  # noqa: SLF001
        info = provider_manager.get_provider_info(provider)
        available = "✅" if info["available"] else "❌"
        platform = info.get("platform", "Unknown")
        description = info.get("description", "N/A")

        providers_table.add_row(provider.value, available, platform, description)

    console.print(providers_table)

    # Provider priority
    priority_table = Table(
        title="Provider Priority",
        show_header=True,
        header_style="bold green",
    )
    priority_table.add_column("Priority", style="cyan")
    priority_table.add_column("Provider", style="white")
    priority_table.add_column("Status", style="white")

    for i, provider in enumerate(provider_manager._provider_priority, 1):  # noqa: SLF001
        info = provider_manager.get_provider_info(provider)
        status = "Available" if info["available"] else "Not Available"
        priority_table.add_row(str(i), provider.value, status)

    console.print(priority_table)
