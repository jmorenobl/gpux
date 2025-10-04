"""Run command for GPUX CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import typer
from rich.console import Console
from rich.json import JSON
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gpux.config.parser import GPUXConfigParser
from gpux.core.runtime import GPUXRuntime

console = Console()
logger = logging.getLogger(__name__)

run_app = typer.Typer(name="run", help="Run inference on models")


@run_app.command()
def run_command(
    model_name: str = typer.Argument(
        ...,
        help="Name of the model to run",
    ),
    input_data: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input data (JSON string or file path with @ prefix)",
    ),
    input_file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Input file path",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path",
    ),
    config_file: str = typer.Option(
        "gpux.yml",
        "--config",
        "-c",
        help="Configuration file name",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Preferred execution provider",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        "-b",
        help="Run benchmark instead of single inference",
    ),
    num_runs: int = typer.Option(
        100,
        "--runs",
        help="Number of benchmark runs",
    ),
    warmup_runs: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup runs",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """Run inference on a model.
    
    This command loads a model and runs inference on the provided input data.
    
    Examples:
        gpux run sentiment-analysis --input '{"text": "I love this!"}'
        gpux run image-classifier --file input.json
        gpux run model-name --benchmark --runs 1000
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find model configuration
        model_path = _find_model_config(model_name, config_file)
        if not model_path:
            console.print(f"[red]Error: Model '{model_name}' not found[/red]")
            raise typer.Exit(1)
        
        # Parse configuration
        parser = GPUXConfigParser()
        config = parser.parse_file(model_path / config_file)
        
        # Get model file path
        model_file = parser.get_model_path(model_path)
        if not model_file or not model_file.exists():
            console.print(f"[red]Error: Model file not found: {model_file}[/red]")
            raise typer.Exit(1)
        
        # Load input data
        input_data_dict = _load_input_data(input_data, input_file)
        if not input_data_dict:
            console.print("[red]Error: No input data provided[/red]")
            raise typer.Exit(1)
        
        # Initialize runtime
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Loading model...", total=None)
            runtime = GPUXRuntime(
                model_path=model_file,
                provider=provider,
                **config.runtime.dict(),
            )
            progress.update(task, completed=100)
        
        # Run inference or benchmark
        if benchmark:
            _run_benchmark(runtime, input_data_dict, num_runs, warmup_runs, output_file)
        else:
            _run_inference(runtime, input_data_dict, output_file)
        
        # Cleanup
        runtime.cleanup()
        
    except Exception as e:
        console.print(f"[red]Run failed: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _find_model_config(model_name: str, config_file: str) -> Optional[Path]:
    """Find model configuration file.
    
    Args:
        model_name: Name of the model
        config_file: Configuration file name
        
    Returns:
        Path to model directory or None if not found
    """
    # Check current directory
    current_dir = Path(".")
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
                import json
                with open(info_file) as f:
                    info = json.load(f)
                if info.get("name") == model_name:
                    return info_file.parent.parent
            except Exception:
                continue
    
    return None


def _load_input_data(input_data: Optional[str], input_file: Optional[str]) -> Optional[dict]:
    """Load input data from various sources.
    
    Args:
        input_data: Input data string or file path
        input_file: Input file path
        
    Returns:
        Input data dictionary or None
    """
    if input_file:
        # Load from file
        try:
            with open(input_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading input file: {e}[/red]")
            return None
    
    if input_data:
        if input_data.startswith("@"):
            # Load from file specified with @ prefix
            file_path = input_data[1:]
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[red]Error loading input file: {e}[/red]")
                return None
        else:
            # Parse JSON string
            try:
                return json.loads(input_data)
            except Exception as e:
                console.print(f"[red]Error parsing input JSON: {e}[/red]")
                return None
    
    return None


def _run_inference(runtime: GPUXRuntime, input_data: dict, output_file: Optional[str]) -> None:
    """Run single inference.
    
    Args:
        runtime: GPUX runtime instance
        input_data: Input data dictionary
        output_file: Output file path
    """
    console.print("[blue]Running inference...[/blue]")
    
    # Convert input data to numpy arrays
    import numpy as np
    numpy_input = {}
    for key, value in input_data.items():
        if isinstance(value, list):
            numpy_input[key] = np.array(value)
        else:
            numpy_input[key] = value
    
    # Run inference
    results = runtime.infer(numpy_input)
    
    # Convert results to JSON-serializable format
    output_data = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):
            output_data[key] = value.tolist()
        else:
            output_data[key] = value
    
    # Display or save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"[green]Results saved to: {output_file}[/green]")
    else:
        console.print(JSON.from_data(output_data))


def _run_benchmark(
    runtime: GPUXRuntime, 
    input_data: dict, 
    num_runs: int, 
    warmup_runs: int,
    output_file: Optional[str]
) -> None:
    """Run benchmark.
    
    Args:
        runtime: GPUX runtime instance
        input_data: Input data dictionary
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        output_file: Output file path
    """
    console.print(f"[blue]Running benchmark with {num_runs} runs (warmup: {warmup_runs})...[/blue]")
    
    # Convert input data to numpy arrays
    import numpy as np
    numpy_input = {}
    for key, value in input_data.items():
        if isinstance(value, list):
            numpy_input[key] = np.array(value)
        else:
            numpy_input[key] = value
    
    # Run benchmark
    metrics = runtime.benchmark(numpy_input, num_runs, warmup_runs)
    
    # Display benchmark results
    benchmark_table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    benchmark_table.add_column("Metric", style="cyan")
    benchmark_table.add_column("Value", style="white")
    
    for key, value in metrics.items():
        if "time" in key:
            benchmark_table.add_row(key.replace("_", " ").title(), f"{value:.2f} ms")
        elif "fps" in key:
            benchmark_table.add_row(key.replace("_", " ").title(), f"{value:.1f}")
        else:
            benchmark_table.add_row(key.replace("_", " ").title(), f"{value:.4f}")
    
    console.print(benchmark_table)
    
    # Save results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"[green]Benchmark results saved to: {output_file}[/green]")
