#!/usr/bin/env python3
"""
Simple Phase 1 Validation Script

This script validates the basic Phase 1 functionality:
- Models can be pulled from Hugging Face
- Models are cached properly
- Basic infrastructure works
"""

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
        execution_time = time.time() - start_time
        return True, result.stdout, execution_time
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        return False, f"Error: {e.stderr}", execution_time
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, "Command timed out", execution_time


def test_model_pull(model_name, description):
    """Test pulling a model."""
    print(f"\n🧪 Testing {description}")
    print(f"   Model: {model_name}")

    # Step 1: Pull model
    print("   📥 Pulling model...")
    pull_success, pull_output, pull_time = run_command(
        ["uv", "run", "gpux", "pull", model_name]
    )

    if not pull_success:
        print(f"   ❌ Pull failed: {pull_output}")
        return False, pull_time, 0

    print(f"   ✅ Pull completed in {pull_time:.2f}s")

    # Step 2: Test inspect (to verify model is cached)
    print("   🔍 Testing model inspection...")
    inspect_success, inspect_output, inspect_time = run_command(
        ["uv", "run", "gpux", "inspect", model_name]
    )

    if not inspect_success:
        print(f"   ❌ Inspect failed: {inspect_output}")
        return False, pull_time, inspect_time

    print(f"   ✅ Inspect completed in {inspect_time:.2f}s")

    total_time = pull_time + inspect_time
    success = pull_success and inspect_success

    print(f"   📊 Total time: {total_time:.2f}s")
    print(f"   {'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}")

    return success, pull_time, total_time


def main():
    """Main function."""
    print("🚀 Simple Phase 1 Validation")
    print("=" * 40)

    # Check if gpux is available
    try:
        subprocess.run(["uv", "run", "gpux", "--help"], check=True, capture_output=True)  # nosec
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: gpux command not found. Please install GPUX first.")
        sys.exit(1)

    # Test models (focus on pull and inspect functionality)
    test_models = [
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "description": "Sentiment Analysis",
        },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Text Embeddings",
        },
        {
            "name": "facebook/opt-125m",
            "description": "Text Generation",
        },
    ]

    results = []

    for model in test_models:
        success, pull_time, total_time = test_model_pull(
            model["name"], model["description"]
        )
        results.append(
            {
                "name": model["name"],
                "success": success,
                "pull_time": pull_time,
                "total_time": total_time,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("SIMPLE VALIDATION SUMMARY")
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

    if success_rate >= 100:
        print("🎉 SIMPLE VALIDATION PASSED!")
        print("   All models successfully pulled and inspected")
        print("   Phase 1 infrastructure is working correctly")
    elif success_rate >= 66:
        print("⚠️  SIMPLE VALIDATION PARTIALLY PASSED")
        print("   Most models work, but some issues detected")
    else:
        print("❌ SIMPLE VALIDATION FAILED")
        print("   Multiple models failed - check implementation")

    return 0 if success_rate >= 66 else 1


if __name__ == "__main__":
    sys.exit(main())
