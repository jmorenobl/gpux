#!/usr/bin/env python3
"""GPUX validation script for final testing and verification."""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpux.config.parser import GPUXConfigParser
from gpux.core.providers import ProviderManager
from gpux.utils.helpers import get_system_info, check_dependencies

console = Console()
app = typer.Typer(name="validate", help="Validate GPUX installation and configuration")


@app.command()
def system() -> None:
    """Validate system requirements and dependencies."""
    console.print("[bold blue]🔍 Validating System Requirements[/bold blue]")
    
    # Check system info
    system_info = get_system_info()
    
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in system_info.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)
    
    # Check dependencies
    console.print("\n[bold blue]📦 Checking Dependencies[/bold blue]")
    deps = check_dependencies()
    
    deps_table = Table(title="Dependencies")
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status", style="white")
    
    for package, available in deps.items():
        status = "✅ Available" if available else "❌ Missing"
        deps_table.add_row(package, status)
    
    console.print(deps_table)
    
    # Check for critical dependencies
    critical_deps = ["onnxruntime", "pydantic", "typer", "rich"]
    missing_critical = [dep for dep in critical_deps if not deps.get(dep, False)]
    
    if missing_critical:
        console.print(f"\n[red]❌ Missing critical dependencies: {', '.join(missing_critical)}[/red]")
        raise typer.Exit(1)
    else:
        console.print("\n[green]✅ All critical dependencies available[/green]")


@app.command()
def providers() -> None:
    """Validate execution providers."""
    console.print("[bold blue]🔍 Validating Execution Providers[/bold blue]")
    
    try:
        manager = ProviderManager()
        available_providers = manager.get_available_providers()
        
        table = Table(title="Available Execution Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Platform", style="white")
        table.add_column("Description", style="white")
        
        for provider in manager._provider_priority:
            info = manager.get_provider_info(provider)
            status = "✅ Available" if provider.value in available_providers else "❌ Not Available"
            table.add_row(
                provider.value,
                status,
                info.get("platform", "Unknown"),
                info.get("description", "No description")
            )
        
        console.print(table)
        
        if available_providers:
            console.print(f"\n[green]✅ Found {len(available_providers)} available providers[/green]")
        else:
            console.print("\n[yellow]⚠️  No GPU providers available, CPU fallback will be used[/yellow]")
            
    except Exception as e:
        console.print(f"\n[red]❌ Error validating providers: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def config(config_file: str = typer.Argument("gpux.yml", help="Configuration file to validate")) -> None:
    """Validate GPUX configuration file."""
    console.print(f"[bold blue]🔍 Validating Configuration: {config_file}[/bold blue]")
    
    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"[red]❌ Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)
    
    try:
        parser = GPUXConfigParser()
        config = parser.parse_file(config_path)
        
        # Display configuration
        table = Table(title="Configuration Details")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Name", config.name)
        table.add_row("Version", config.version)
        table.add_row("Model Source", str(config.model.source))
        table.add_row("Model Format", config.model.format)
        table.add_row("GPU Memory", config.runtime.gpu.memory)
        table.add_row("GPU Backend", config.runtime.gpu.backend)
        table.add_row("Batch Size", str(config.runtime.batch_size))
        table.add_row("Timeout", str(config.runtime.timeout))
        
        console.print(table)
        
        # Validate model path
        if parser.validate_model_path(Path(".")):
            console.print("\n[green]✅ Model file found and valid[/green]")
        else:
            console.print("\n[yellow]⚠️  Model file not found or invalid[/yellow]")
        
        console.print("\n[green]✅ Configuration is valid[/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def examples() -> None:
    """Validate example configurations."""
    console.print("[bold blue]🔍 Validating Example Configurations[/bold blue]")
    
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        console.print("[red]❌ Examples directory not found[/red]")
        raise typer.Exit(1)
    
    example_configs = list(examples_dir.glob("*/gpux.yml"))
    
    if not example_configs:
        console.print("[red]❌ No example configurations found[/red]")
        raise typer.Exit(1)
    
    table = Table(title="Example Configurations")
    table.add_column("Example", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Model File", style="white")
    
    all_valid = True
    
    for config_file in example_configs:
        example_name = config_file.parent.name
        
        try:
            parser = GPUXConfigParser()
            config = parser.parse_file(config_file)
            
            # Check if model file exists
            model_path = parser.get_model_path(config_file.parent)
            model_exists = model_path and model_path.exists() if model_path else False
            
            status = "✅ Valid" if model_exists else "⚠️  No Model"
            model_status = "Found" if model_exists else "Missing"
            
            table.add_row(example_name, status, model_status)
            
            if not model_exists:
                all_valid = False
                
        except Exception as e:
            table.add_row(example_name, "❌ Error", str(e)[:50])
            all_valid = False
    
    console.print(table)
    
    if all_valid:
        console.print("\n[green]✅ All example configurations are valid[/green]")
    else:
        console.print("\n[yellow]⚠️  Some examples have issues (missing model files are expected)[/yellow]")


@app.command()
def all() -> None:
    """Run all validation checks."""
    console.print("[bold green]🚀 Running Complete GPUX Validation[/bold green]\n")
    
    try:
        # System validation
        system()
        console.print()
        
        # Providers validation
        providers()
        console.print()
        
        # Examples validation
        examples()
        console.print()
        
        console.print("[bold green]🎉 All validation checks completed successfully![/bold green]")
        console.print("\n[bold blue]GPUX is ready to use! 🚀[/bold blue]")
        
    except typer.Exit as e:
        console.print(f"\n[red]❌ Validation failed with exit code {e.exit_code}[/red]")
        raise
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error during validation: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
