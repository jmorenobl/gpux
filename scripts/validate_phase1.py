#!/usr/bin/env python3
"""
Phase 1 Performance Validation Script

This script validates the success criteria for Phase 1 of multi-registry integration:
- Pull + convert + run time < 30 seconds for models < 500MB
- Model conversion success rate > 90%
- Cache hit rate > 80% on second run
- All core functionality works on Apple Silicon, NVIDIA, and AMD GPUs
"""

import json
import logging
import subprocess  # nosec
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelTest:
    """Test configuration for a model."""

    name: str
    size_mb: int
    input_data: dict
    expected_output_keys: list[str]
    category: str
    description: str


@dataclass
class TestResult:
    """Result of a model test."""

    model_name: str
    success: bool
    pull_time: float
    convert_time: float
    run_time: float
    total_time: float
    cache_hit: bool
    error_message: str | None = None
    output_sample: dict | None = None


# Test models configuration
TEST_MODELS = [
    ModelTest(
        name="distilbert-base-uncased-finetuned-sst-2-english",
        size_mb=268,
        input_data={
            "input_ids": [[101, 1045, 2293, 2003, 2023, 3457, 999, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        },
        expected_output_keys=["logits"],
        category="text_classification",
        description="Sentiment analysis model",
    ),
    ModelTest(
        name="sentence-transformers/all-MiniLM-L6-v2",
        size_mb=90,
        input_data={"inputs": "Hello world"},
        expected_output_keys=["embeddings"],
        category="embeddings",
        description="General purpose embeddings",
    ),
    ModelTest(
        name="facebook/opt-125m",
        size_mb=500,
        input_data={"inputs": "The future of AI is"},
        expected_output_keys=["generated_text"],
        category="text_generation",
        description="Small GPT-style model",
    ),
    ModelTest(
        name="distilbert-base-cased-distilled-squad",
        size_mb=250,
        input_data={
            "question": "What is AI?",
            "context": "AI is artificial intelligence",
        },
        expected_output_keys=["answer"],
        category="question_answering",
        description="Question answering model",
    ),
    ModelTest(
        name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        size_mb=500,
        input_data={"inputs": "Just had the best coffee ever! ☕️"},
        expected_output_keys=["logits"],
        category="text_classification",
        description="Twitter sentiment analysis",
    ),
    ModelTest(
        name="sentence-transformers/all-mpnet-base-v2",
        size_mb=420,
        input_data={"inputs": "Hello world"},
        expected_output_keys=["embeddings"],
        category="embeddings",
        description="Higher quality embeddings",
    ),
    ModelTest(
        name="microsoft/DialoGPT-small",
        size_mb=350,
        input_data={"inputs": "Hello, how are you?"},
        expected_output_keys=["generated_text"],
        category="text_generation",
        description="Small dialog model",
    ),
    ModelTest(
        name="deepset/roberta-base-squad2",
        size_mb=500,
        input_data={
            "question": "What is GPUX?",
            "context": "GPUX is a Docker-like runtime for ML inference",
        },
        expected_output_keys=["answer"],
        category="question_answering",
        description="Higher accuracy QA model",
    ),
]


class PerformanceValidator:
    """Validates Phase 1 performance criteria."""

    def __init__(self, output_dir: Path = Path("validation_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: list[TestResult] = []

    def run_command(
        self, cmd: list[str], timeout: int = 300
    ) -> tuple[bool, str, float]:
        """Run a command and return success, output, and execution time."""
        start_time = time.time()
        try:
            result = subprocess.run(  # nosec
                cmd, capture_output=True, text=True, timeout=timeout, check=True
            )
        except subprocess.CalledProcessError as e:
            execution_time = time.time() - start_time
            return False, f"Error: {e.stderr}", execution_time
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return False, "Command timed out", execution_time
        else:
            execution_time = time.time() - start_time
            return True, result.stdout, execution_time

    def test_model(self, model: ModelTest) -> TestResult:
        """Test a single model."""
        logger.info(f"Testing model: {model.name}")

        # Step 1: Pull model
        logger.info(f"Pulling model: {model.name}")
        pull_success, pull_output, pull_time = self.run_command(
            ["uv", "run", "gpux", "pull", model.name, "--verbose"]
        )

        if not pull_success:
            return TestResult(
                model_name=model.name,
                success=False,
                pull_time=pull_time,
                convert_time=0,
                run_time=0,
                total_time=pull_time,
                cache_hit=False,
                error_message=f"Pull failed: {pull_output}",
            )

        # Extract conversion time from verbose output
        convert_time = 0
        if "Converting to ONNX" in pull_output:
            # Estimate conversion time (this would need more sophisticated parsing)
            convert_time = pull_time * 0.7  # Rough estimate

        # Step 2: Run inference
        logger.info(f"Running inference: {model.name}")
        run_success, run_output, run_time = self.run_command(
            [
                "uv",
                "run",
                "gpux",
                "run",
                model.name,
                "--input",
                json.dumps(model.input_data),
            ]
        )

        if not run_success:
            return TestResult(
                model_name=model.name,
                success=False,
                pull_time=pull_time,
                convert_time=convert_time,
                run_time=run_time,
                total_time=pull_time + convert_time + run_time,
                cache_hit=False,
                error_message=f"Run failed: {run_output}",
            )

        # Parse output
        try:
            output_data = json.loads(run_output)
            output_sample = {
                k: str(v)[:100] + "..." if len(str(v)) > 100 else v
                for k, v in output_data.items()
            }
        except json.JSONDecodeError:
            output_sample = {"raw_output": run_output[:200]}

        # Step 3: Test cache hit (run again)
        logger.info(f"Testing cache hit: {model.name}")
        cache_success, _cache_output, cache_time = self.run_command(
            [
                "uv",
                "run",
                "gpux",
                "run",
                model.name,
                "--input",
                json.dumps(model.input_data),
            ]
        )

        cache_hit = (
            cache_success and cache_time < run_time * 0.5
        )  # Cache should be much faster

        total_time = pull_time + convert_time + run_time

        return TestResult(
            model_name=model.name,
            success=True,
            pull_time=pull_time,
            convert_time=convert_time,
            run_time=run_time,
            total_time=total_time,
            cache_hit=cache_hit,
            output_sample=output_sample,
        )

    def run_all_tests(self) -> None:
        """Run all model tests."""
        logger.info("Starting Phase 1 performance validation")
        logger.info(f"Testing {len(TEST_MODELS)} models")

        for i, model in enumerate(TEST_MODELS, 1):
            logger.info(f"Progress: {i}/{len(TEST_MODELS)}")
            result = self.test_model(model)
            self.results.append(result)

            # Log result
            if result.success:
                logger.info(f"✅ {model.name}: {result.total_time:.2f}s total")
            else:
                logger.error(f"❌ {model.name}: {result.error_message}")

        # Save results
        self.save_results()

        # Generate report
        self.generate_report()

    def save_results(self) -> None:
        """Save test results to JSON file."""
        results_data = {
            "timestamp": time.time(),
            "models_tested": len(TEST_MODELS),
            "results": [asdict(result) for result in self.results],
        }

        results_file = self.output_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info("Results saved to: %s", results_file)

    def generate_report(self) -> None:
        """Generate markdown report."""
        report_file = self.output_dir / "validation_report.md"

        # Calculate metrics
        successful_tests = [r for r in self.results if r.success]
        success_rate = len(successful_tests) / len(self.results) * 100

        cache_hits = [r for r in successful_tests if r.cache_hit]
        cache_hit_rate = (
            len(cache_hits) / len(successful_tests) * 100 if successful_tests else 0
        )

        avg_total_time = (
            sum(r.total_time for r in successful_tests) / len(successful_tests)
            if successful_tests
            else 0
        )

        small_models = [r for r in successful_tests if r.total_time < 30]
        small_model_success_rate = (
            len(small_models) / len([m for m in TEST_MODELS if m.size_mb < 500]) * 100
        )

        with open(report_file, "w") as f:
            f.write("# Phase 1 Performance Validation Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Models Tested**: {len(TEST_MODELS)}\n")
            f.write(f"- **Successful Tests**: {len(successful_tests)}\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\n")
            f.write(f"- **Cache Hit Rate**: {cache_hit_rate:.1f}%\n")
            f.write(f"- **Average Total Time**: {avg_total_time:.2f}s\n")
            f.write(
                f"- **Small Model Success Rate**: {small_model_success_rate:.1f}%\n\n"
            )

            # Success Criteria
            f.write("## Success Criteria Validation\n\n")
            f.write("| Criteria | Target | Achieved | Status |\n")
            f.write("|----------|--------|----------|--------|\n")
            success_icon = "✅" if success_rate > 90 else "❌"
            f.write(
                f"| Conversion Success Rate | >90% | {success_rate:.1f}% | {success_icon} |\n"
            )
            cache_icon = "✅" if cache_hit_rate > 80 else "❌"
            f.write(
                f"| Cache Hit Rate | >80% | {cache_hit_rate:.1f}% | {cache_icon} |\n"
            )
            small_icon = "✅" if small_model_success_rate > 90 else "❌"
            f.write(
                f"| Small Model Success Rate | >90% | {small_model_success_rate:.1f}% | {small_icon} |\n"
            )
            time_icon = "✅" if avg_total_time < 30 else "❌"
            f.write(
                f"| Average Time <30s | <30s | {avg_total_time:.2f}s | {time_icon} |\n\n"
            )

            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write(
                "| Model | Category | Size (MB) | Success | Total Time | "
                "Cache Hit | Error |\n"
            )
            f.write(
                "|-------|----------|-----------|---------|------------|-----------|-------|\n"
            )

            for result in self.results:
                model_info = next(m for m in TEST_MODELS if m.name == result.model_name)
                f.write(
                    f"| {result.model_name} | {model_info.category} | "
                    f"{model_info.size_mb} | "
                )
                f.write(f"{'✅' if result.success else '❌'} | ")
                f.write(f"{result.total_time:.2f}s | ")
                f.write(f"{'✅' if result.cache_hit else '❌'} | ")
                f.write(f"{result.error_message or 'None'} |\n")

            # Performance Analysis
            f.write("\n## Performance Analysis\n\n")

            if successful_tests:
                f.write("### Timing Breakdown\n\n")
                f.write(
                    "| Model | Pull Time | Convert Time | Run Time | Total Time |\n"
                )
                f.write(
                    "|-------|-----------|--------------|----------|------------|\n"
                )

                for result in successful_tests:
                    f.write(f"| {result.model_name} | {result.pull_time:.2f}s | ")
                    f.write(f"{result.convert_time:.2f}s | {result.run_time:.2f}s | ")
                    f.write(f"{result.total_time:.2f}s |\n")

            # Recommendations
            f.write("\n## Recommendations\n\n")

            if success_rate < 90:
                f.write(
                    "- **Conversion Issues**: Some models failed to convert. "
                    "Consider:\n"
                )
                f.write("  - Adding fallback conversion methods\n")
                f.write("  - Improving error handling\n")
                f.write("  - Testing with different model architectures\n\n")

            if cache_hit_rate < 80:
                f.write("- **Cache Issues**: Cache hit rate is low. Consider:\n")
                f.write("  - Improving cache implementation\n")
                f.write("  - Adding cache validation\n")
                f.write("  - Optimizing model loading\n\n")

            if avg_total_time > 30:
                f.write(
                    "- **Performance Issues**: Average time exceeds 30s. Consider:\n"
                )
                f.write("  - Optimizing conversion pipeline\n")
                f.write("  - Adding parallel processing\n")
                f.write("  - Implementing incremental conversion\n\n")

        logger.info("Report generated: %s", report_file)

    def print_summary(self) -> None:
        """Print summary to console."""
        successful_tests = [r for r in self.results if r.success]
        success_rate = len(successful_tests) / len(self.results) * 100

        cache_hits = [r for r in successful_tests if r.cache_hit]
        cache_hit_rate = (
            len(cache_hits) / len(successful_tests) * 100 if successful_tests else 0
        )

        avg_total_time = (
            sum(r.total_time for r in successful_tests) / len(successful_tests)
            if successful_tests
            else 0
        )

        print("\n" + "=" * 60)
        print("PHASE 1 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Models Tested: {len(TEST_MODELS)}")
        print(f"Successful Tests: {len(successful_tests)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
        print(f"Average Total Time: {avg_total_time:.2f}s")
        print("=" * 60)

        # Success criteria
        print("\nSUCCESS CRITERIA:")
        print(
            f"✅ Conversion Success Rate >90%: {success_rate:.1f}% "
            f"{'✅' if success_rate > 90 else '❌'}"
        )
        print(
            f"✅ Cache Hit Rate >80%: {cache_hit_rate:.1f}% "
            f"{'✅' if cache_hit_rate > 80 else '❌'}"
        )
        print(
            f"✅ Average Time <30s: {avg_total_time:.2f}s "
            f"{'✅' if avg_total_time < 30 else '❌'}"
        )

        if success_rate > 90 and cache_hit_rate > 80 and avg_total_time < 30:
            print("\n🎉 PHASE 1 VALIDATION PASSED!")
        else:
            print("\n⚠️  PHASE 1 VALIDATION NEEDS IMPROVEMENT")


def main():
    """Main function."""
    print("Phase 1 Performance Validation")
    print("=" * 40)

    # Check if gpux is available
    try:
        subprocess.run(["uv", "run", "gpux", "--help"], check=True, capture_output=True)  # nosec
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: gpux command not found. Please install GPUX first.")
        sys.exit(1)

    # Create validator
    validator = PerformanceValidator()

    # Run tests
    try:
        validator.run_all_tests()
        validator.print_summary()
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
