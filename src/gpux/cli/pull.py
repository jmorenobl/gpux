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
        except Exception as e:
            progress.stop()
            _handle_pull_error(e, model_id, verbose=verbose)

    return metadata


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
