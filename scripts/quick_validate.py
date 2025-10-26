#!/usr/bin/env python3
"""
Quick Phase 1 Validation Script

This script runs a quick validation with 3 popular models to verify basic functionality.
"""

import json
import subprocess  # nosec
import sys
import time


def run_command(cmd, timeout=120):
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


def test_model(model_name, input_data, description):
    """Test a single model."""
    print(f"\n🧪 Testing {description}")
    print(f"   Model: {model_name}")

    # Step 1: Pull model
    print("   📥 Pulling model...")
    pull_success, pull_output, pull_time = run_command(
        ["uv", "run", "gpux", "pull", model_name]
    )

    if not pull_success:
        print(f"   ❌ Pull failed: {pull_output}")
        return False, 0, 0, 0

    print(f"   ✅ Pull completed in {pull_time:.2f}s")

    # Step 2: Run inference
    print("   🚀 Running inference...")
    run_success, run_output, run_time = run_command(
        ["uv", "run", "gpux", "run", model_name, "--input", json.dumps(input_data)]
    )

    if not run_success:
        print(f"   ❌ Run failed: {run_output}")
        return False, pull_time, 0, run_time

    print(f"   ✅ Inference completed in {run_time:.2f}s")

    # Step 3: Test cache hit
    print("   💾 Testing cache hit...")
    cache_success, _cache_output, cache_time = run_command(
        ["uv", "run", "gpux", "run", model_name, "--input", json.dumps(input_data)]
    )

    cache_hit = cache_success and cache_time < run_time * 0.5
    print(f"   {'✅' if cache_hit else '❌'} Cache hit: {cache_time:.2f}s")

    total_time = pull_time + run_time
    success = pull_success and run_success

    print(f"   📊 Total time: {total_time:.2f}s")
    print(f"   {'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}")

    return success, pull_time, run_time, total_time


def main():
    """Main function."""
    print("🚀 Quick Phase 1 Validation")
    print("=" * 40)

    # Check if gpux is available
    try:
        subprocess.run(["uv", "run", "gpux", "--help"], check=True, capture_output=True)  # nosec
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: gpux command not found. Please install GPUX first.")
        sys.exit(1)

    # Test models
    test_models = [
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "input": {"inputs": "I love this product!"},
            "description": "Sentiment Analysis",
        },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "input": {"inputs": "Hello world"},
            "description": "Text Embeddings",
        },
        {
            "name": "facebook/opt-125m",
            "input": {"inputs": "The future of AI is"},
            "description": "Text Generation",
        },
    ]

    results = []

    for model in test_models:
        success, pull_time, run_time, total_time = test_model(
            model["name"], model["input"], model["description"]
        )
        results.append(
            {
                "name": model["name"],
                "success": success,
                "pull_time": pull_time,
                "run_time": run_time,
                "total_time": total_time,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("QUICK VALIDATION SUMMARY")
    print("=" * 60)

    successful_tests = [r for r in results if r["success"]]
    success_rate = len(successful_tests) / len(results) * 100
    avg_time = (
        sum(r["total_time"] for r in successful_tests) / len(successful_tests)
        if successful_tests
        else 0
    )

    print(f"Models Tested: {len(results)}")
    print(f"Successful Tests: {len(successful_tests)}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Time: {avg_time:.2f}s")

    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {result['name']}: {status} ({result['total_time']:.2f}s)")

    print("=" * 60)

    if success_rate >= 100 and avg_time < 30:
        print("🎉 QUICK VALIDATION PASSED!")
        print("   All models successfully pulled and run in under 30s")
    elif success_rate >= 66:
        print("⚠️  QUICK VALIDATION PARTIALLY PASSED")
        print("   Most models work, but some issues detected")
    else:
        print("❌ QUICK VALIDATION FAILED")
        print("   Multiple models failed - check implementation")

    return 0 if success_rate >= 66 else 1


if __name__ == "__main__":
    sys.exit(main())
