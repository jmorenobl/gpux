"""GPUX pull command for downloading models from registries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gpux.core.conversion import ConfigGenerator, PyTorchConverter, TensorFlowConverter
from gpux.core.managers import (
    HuggingFaceManager,
    ModelManager,
    ModelMetadata,
    RegistryConfig,
)
from gpux.core.managers.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    NetworkError,
    RegistryError,
)

logger = logging.getLogger(__name__)
console = Console()

# Registry configuration mapping
REGISTRY_CONFIGS = {
    "huggingface": RegistryConfig(
        name="huggingface",
        api_url="https://huggingface.co",
        auth_token=None,  # Will be read from environment
    ),
    "hf": RegistryConfig(
        name="huggingface",
        api_url="https://huggingface.co",
        auth_token=None,  # Will be read from environment
    ),
}

# Default registry
DEFAULT_REGISTRY = "huggingface"


def pull_command(
    model_id: Annotated[
        str, typer.Argument(help="Model identifier (e.g., 'microsoft/DialoGPT-medium')")
    ],
    registry: Annotated[
        str, typer.Option("--registry", "-r", help="Registry to pull from")
    ] = DEFAULT_REGISTRY,
    revision: Annotated[
        str, typer.Option("--revision", help="Model revision/branch to pull")
    ] = "main",
    cache_dir: Annotated[
        str | None, typer.Option("--cache-dir", help="Custom cache directory")
    ] = None,
    *,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force re-download even if model exists"),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose output")
    ] = False,
) -> None:
    """Pull a model from a supported registry.

    This command downloads models from registries like Hugging Face Hub and caches
    them locally for use with GPUX. The downloaded models are automatically converted
    to ONNX format and configured for optimal inference.

    Examples:
        gpux pull microsoft/DialoGPT-medium
        gpux pull microsoft/DialoGPT-medium --revision main
        gpux pull microsoft/DialoGPT-medium --registry huggingface --cache-dir ~/models
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        _validate_registry(registry)
        config = REGISTRY_CONFIGS[registry]
        manager = _create_model_manager(config)
        cache_path = Path(cache_dir) if cache_dir else None

        _display_pull_info(model_id, registry, revision, cache_path)
        metadata = _pull_model_with_progress(
            manager, model_id, revision, cache_path, force=force, verbose=verbose
        )
        _display_success_info(metadata)

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during model pull.")
        console.print(f"[red]Error:[/red] An unexpected error occurred: {e}")
        if verbose:
            console.print(f"[dim]Details: {e}[/dim]")
        raise typer.Exit(1) from e


def _validate_registry(registry: str) -> None:
    """Validate that the registry is supported."""
    if registry not in REGISTRY_CONFIGS:
        supported_registries = ", ".join(REGISTRY_CONFIGS.keys())
        console.print(
            f"[red]Error:[/red] Unsupported registry '{registry}'. "
            f"Supported registries: {supported_registries}"
        )
        raise typer.Exit(1)


def _pull_model_with_progress(
    manager: ModelManager,
    model_id: str,
    revision: str,
    cache_path: Path | None,
    *,
    force: bool,
    verbose: bool,
) -> ModelMetadata:
    """Pull model with progress indicator and error handling."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Pulling {model_id}...", total=None)

        try:
            metadata = manager.pull_model(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_path,
                force_download=force,
            )
            progress.update(task, description=f"Successfully pulled {model_id}")

            # Auto-convert to ONNX if needed
            if metadata.format in ("pytorch", "tensorflow"):
                progress.update(task, description=f"Converting {model_id} to ONNX...")
                try:
                    onnx_path = _convert_model_to_onnx(metadata, cache_path)
                    progress.update(
                        task, description=f"Generating config for {model_id}"
                    )
                    _generate_config_file(metadata, onnx_path, cache_path)
                    progress.update(
                        task,
                        description=f"Successfully converted and configured {model_id}",
                    )
                except Exception as conv_error:
                    if verbose:
                        console.print(
                            f"[yellow]Warning:[/yellow] Conversion failed: {conv_error}"
                        )
                        import traceback

                        traceback.print_exc()
                    else:
                        console.print(
                            "[yellow]Warning:[/yellow] Model conversion failed, "
                            "but model is available for manual conversion"
                        )

        except Exception as e:
            progress.stop()
            _handle_pull_error(e, model_id, verbose=verbose)

    return metadata


def _convert_model_to_onnx(metadata: ModelMetadata, cache_path: Path | None) -> Path:
    """Convert model to ONNX format.

    Args:
        metadata: Model metadata
        cache_path: Cache directory

    Returns:
        Path to converted ONNX model
    """
    converter: PyTorchConverter | TensorFlowConverter
    if metadata.format == "pytorch":
        converter = PyTorchConverter(cache_dir=cache_path)
    elif metadata.format == "tensorflow":
        converter = TensorFlowConverter(cache_dir=cache_path)
    else:
        msg = f"Unsupported format: {metadata.format}"
        raise ValueError(msg)

    if not converter.can_convert(metadata):
        msg = f"Cannot convert {metadata.format} model"
        raise ValueError(msg)

    return converter.convert(metadata)


def _generate_config_file(
    metadata: ModelMetadata, onnx_path: Path, cache_path: Path | None
) -> None:
    """Generate gpux.yml configuration file.

    Args:
        metadata: Model metadata
        onnx_path: Path to ONNX model
        cache_path: Cache directory
    """
    config_generator = ConfigGenerator()

    # Generate config in the model's directory
    if cache_path is None:
        cache_path = Path.home() / ".gpux" / "models"

    model_dir = (
        cache_path / "huggingface" / metadata.model_id.replace("/", "--") / "main"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    config_path = model_dir / "gpux.yml"
    config_generator.generate_config(metadata, onnx_path, config_path)

    # Update the config file to use absolute path for the ONNX model
    import yaml

    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    config_data["model"]["source"] = str(onnx_path)

    with config_path.open("w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated configuration file: %s", config_path)


def _handle_pull_error(error: Exception, model_id: str, *, verbose: bool) -> None:
    """Handle pull errors with appropriate user messages."""
    if isinstance(error, ModelNotFoundError):
        console.print(f"[red]Error:[/red] Model '{model_id}' not found in registry")
    elif isinstance(error, AuthenticationError):
        console.print("[red]Error:[/red] Authentication failed")
        console.print(
            "[yellow]Hint:[/yellow] Set your authentication token using "
            "environment variables:"
        )
        console.print("  export HF_TOKEN=your_token_here")
    elif isinstance(error, NetworkError):
        console.print("[red]Error:[/red] Network error while pulling model")
        console.print(
            "[yellow]Hint:[/yellow] Check your internet connection and try again"
        )
    elif isinstance(error, RegistryError):
        console.print(f"[red]Error:[/red] Registry error: {error}")
    else:
        console.print(f"[red]Error:[/red] Unexpected error: {error}")

    if verbose:
        console.print(f"[dim]Details: {error}[/dim]")
    raise typer.Exit(1) from error


def _create_model_manager(config: RegistryConfig) -> ModelManager:
    """Create a model manager for the given registry configuration."""
    if config.name == "huggingface":
        return HuggingFaceManager(config)
    msg = f"Unsupported registry: {config.name}"
    raise ValueError(msg)


def _display_pull_info(
    model_id: str, registry: str, revision: str, cache_path: Path | None
) -> None:
    """Display information about the pull operation."""
    info_table = Table(title="Pull Information", show_header=False, box=None)
    info_table.add_column("Field", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="white")

    info_table.add_row("Model ID", model_id)
    info_table.add_row("Registry", registry)
    info_table.add_row("Revision", revision)
    if cache_path:
        info_table.add_row("Cache Directory", str(cache_path))
    else:
        info_table.add_row("Cache Directory", "Default (~/.gpux/models/)")

    console.print(info_table)
    console.print()


def _display_success_info(metadata: ModelMetadata) -> None:
    """Display success information after pulling a model."""
    size_display = (
        _format_size(metadata.size_bytes) if metadata.size_bytes else "Unknown"
    )
    success_panel = Panel(
        f"[green]✓[/green] Successfully pulled model '{metadata.model_id}'\n"
        f"[dim]Registry:[/dim] {metadata.registry}\n"
        f"[dim]Format:[/dim] {metadata.format}\n"
        f"[dim]Size:[/dim] {size_display}\n"
        f"[dim]Files:[/dim] {len(metadata.files)} files downloaded",
        title="[green]Pull Complete[/green]",
        border_style="green",
    )
    console.print(success_panel)

    # Display next steps
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print(f"• Run inference: [cyan]gpux run {metadata.model_id}[/cyan]")
    console.print(f"• Start server: [cyan]gpux serve {metadata.model_id}[/cyan]")
    console.print(f"• Inspect model: [cyan]gpux inspect {metadata.model_id}[/cyan]")


def _format_size(size_bytes: int | None) -> str:
    """Format size in bytes to human-readable format."""
    if size_bytes is None:
        return "Unknown"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes = int(size_bytes / 1024.0)

    return f"{size_bytes:.1f} PB"


if __name__ == "__main__":
    typer.run(pull_command)
