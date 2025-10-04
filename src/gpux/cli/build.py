"""Build command for GPUX CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from gpux.config.parser import GPUXConfigParser
from gpux.core.models import ModelInspector
from gpux.core.providers import ProviderManager

console = Console()
logger = logging.getLogger(__name__)

build_app = typer.Typer(name="build", help="Build and optimize models for GPU inference")


@build_app.command()
def build_command(
    path: str = typer.Argument(
        ".",
        help="Path to the GPUX project directory",
    ),
    config_file: str = typer.Option(
        "gpux.yml",
        "--config",
        "-c",
        help="Configuration file name",
    ),
    optimize: bool = typer.Option(
        True,
        "--optimize/--no-optimize",
        help="Enable model optimization",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Preferred execution provider (cuda, coreml, rocm, etc.)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Build and optimize models for GPU inference.
    
    This command validates the GPUX configuration, inspects the model,
    and prepares it for optimal GPU inference.
    
    Examples:
        gpux build .                    # Build from current directory
        gpux build ./my-model --provider cuda  # Build with specific provider
        gpux build . --no-optimize     # Build without optimization
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_path = Path(path).resolve()
    config_path = project_path / config_file
    
    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Parse configuration
            task1 = progress.add_task("Parsing configuration...", total=100)
            parser = GPUXConfigParser()
            config = parser.parse_file(config_path)
            progress.update(task1, completed=100)
            
            # Validate model path
            task2 = progress.add_task("Validating model path...", total=100)
            if not parser.validate_model_path(project_path):
                console.print("[red]Error: Model file not found[/red]")
                raise typer.Exit(1)
            progress.update(task2, completed=100)
            
            # Inspect model
            task3 = progress.add_task("Inspecting model...", total=100)
            model_path = parser.get_model_path(project_path)
            if not model_path:
                console.print("[red]Error: Could not resolve model path[/red]")
                raise typer.Exit(1)
            
            inspector = ModelInspector()
            model_info = inspector.inspect(model_path)
            progress.update(task3, completed=100)
            
            # Check provider compatibility
            task4 = progress.add_task("Checking provider compatibility...", total=100)
            provider_manager = ProviderManager()
            selected_provider = provider_manager.get_best_provider(provider)
            progress.update(task4, completed=100)
            
            # Optimize model (placeholder for future optimization)
            if optimize:
                task5 = progress.add_task("Optimizing model...", total=100)
                # TODO: Implement model optimization
                progress.update(task5, completed=100)
            
            # Save build artifacts
            task6 = progress.add_task("Saving build artifacts...", total=100)
            build_dir = project_path / ".gpux"
            build_dir.mkdir(exist_ok=True)
            
            # Save model info
            model_info.save(build_dir / "model_info.json")
            
            # Save provider info
            provider_info = provider_manager.get_provider_info(selected_provider)
            import json
            with open(build_dir / "provider_info.json", "w") as f:
                json.dump(provider_info, f, indent=2)
            
            progress.update(task6, completed=100)
        
        # Display build results
        _display_build_results(config, model_info, selected_provider, provider_info)
        
        console.print(f"\n[green]✅ Build completed successfully![/green]")
        console.print(f"[dim]Build artifacts saved to: {build_dir}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_build_results(config, model_info, provider, provider_info) -> None:
    """Display build results in a formatted table."""
    
    # Model information table
    model_table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    model_table.add_column("Property", style="cyan")
    model_table.add_column("Value", style="white")
    
    model_table.add_row("Name", model_info.name)
    model_table.add_row("Version", model_info.version)
    model_table.add_row("Format", model_info.format)
    model_table.add_row("Size", f"{model_info.size_bytes / (1024*1024):.1f} MB")
    model_table.add_row("Inputs", str(len(model_info.inputs)))
    model_table.add_row("Outputs", str(len(model_info.outputs)))
    
    console.print(model_table)
    
    # Provider information table
    provider_table = Table(title="Execution Provider", show_header=True, header_style="bold magenta")
    provider_table.add_column("Property", style="cyan")
    provider_table.add_column("Value", style="white")
    
    provider_table.add_row("Provider", provider.value)
    provider_table.add_row("Platform", provider_info.get("platform", "Unknown"))
    provider_table.add_row("Available", "✅ Yes" if provider_info.get("available", False) else "❌ No")
    provider_table.add_row("Description", provider_info.get("description", "N/A"))
    
    console.print(provider_table)
    
    # Input/Output details
    if model_info.inputs:
        inputs_table = Table(title="Input Specifications", show_header=True, header_style="bold green")
        inputs_table.add_column("Name", style="cyan")
        inputs_table.add_column("Type", style="white")
        inputs_table.add_column("Shape", style="white")
        inputs_table.add_column("Required", style="white")
        
        for inp in model_info.inputs:
            inputs_table.add_row(
                inp.name,
                inp.type,
                str(inp.shape) if inp.shape else "Dynamic",
                "✅" if inp.required else "❌"
            )
        
        console.print(inputs_table)
    
    if model_info.outputs:
        outputs_table = Table(title="Output Specifications", show_header=True, header_style="bold green")
        outputs_table.add_column("Name", style="cyan")
        outputs_table.add_column("Type", style="white")
        outputs_table.add_column("Shape", style="white")
        
        for out in model_info.outputs:
            outputs_table.add_row(
                out.name,
                out.type,
                str(out.shape) if out.shape else "Dynamic"
            )
        
        console.print(outputs_table)
