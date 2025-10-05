# `gpux run`

Run inference on models.

---

## Overview

The `gpux run` command loads a model and runs inference on provided input data. It supports both single inference and benchmarking modes.

```bash
gpux run MODEL_NAME [OPTIONS]
```

---

## Arguments

### `MODEL_NAME` *(required)*

Name of the model to run.

- **Type**: `string`
- **Required**: Yes

**Examples**:
```bash
gpux run sentiment-analysis
gpux run image-classifier
gpux run ./models/bert
```

The command searches for models in:
1. Current directory (if `gpux.yml` exists)
2. Directory specified by model name
3. `.gpux/` directory (for built models)

---

## Options

### Input Options

#### `--input`, `-i`

Input data as JSON string or file path (with `@` prefix).

- **Type**: `string`

```bash
# JSON string
gpux run sentiment --input '{"text": "I love this!"}'

# File path with @ prefix
gpux run sentiment --input @input.json
```

#### `--file`, `-f`

Input file path (alternative to `--input`).

- **Type**: `string`

```bash
gpux run sentiment --file input.json
```

### Output Options

#### `--output`, `-o`

Save output to file.

- **Type**: `string`

```bash
gpux run sentiment --input '{"text": "Great!"}' --output result.json
```

### Configuration Options

#### `--config`, `-c`

Configuration file name.

- **Type**: `string`
- **Default**: `gpux.yml`

```bash
gpux run sentiment --config custom.yml
```

#### `--provider`, `-p`

Preferred execution provider.

- **Type**: `string`
- **Choices**: `cuda`, `coreml`, `rocm`, `directml`, `openvino`, `tensorrt`, `cpu`

```bash
gpux run sentiment --provider cuda
```

### Benchmark Options

#### `--benchmark`

Run benchmark instead of single inference.

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux run sentiment --input '{"text": "Test"}' --benchmark
```

#### `--runs`

Number of benchmark runs.

- **Type**: `integer`
- **Default**: `100`

```bash
gpux run sentiment --input '{"text": "Test"}' --benchmark --runs 1000
```

#### `--warmup`

Number of warmup runs before benchmarking.

- **Type**: `integer`
- **Default**: `10`

```bash
gpux run sentiment --input '{"text": "Test"}' --benchmark --warmup 20
```

### Other Options

#### `--verbose`

Enable verbose output.

- **Type**: `boolean`
- **Default**: `false`

```bash
gpux run sentiment --input '{"text": "Test"}' --verbose
```

---

## Input Formats

### JSON String

Pass input data directly as a JSON string:

```bash
gpux run sentiment --input '{"text": "I love GPUX!"}'
```

### JSON File

Load input from a JSON file:

```bash
gpux run sentiment --file input.json
```

**input.json**:
```json
{
  "text": "I love GPUX!"
}
```

### File Path with @ Prefix

Alternative file loading syntax:

```bash
gpux run sentiment --input @input.json
```

### Multiple Inputs

For models with multiple inputs:

```json
{
  "input_ids": [1, 2, 3, 4, 5],
  "attention_mask": [1, 1, 1, 1, 1]
}
```

---

## Output Formats

### Console Output (Default)

Results are displayed as formatted JSON:

```json
{
  "sentiment": [0.1, 0.9]
}
```

### File Output

Save results to a file:

```bash
gpux run sentiment --input '{"text": "Great!"}' --output result.json
```

**result.json**:
```json
{
  "sentiment": [0.1, 0.9],
  "labels": ["negative", "positive"]
}
```

---

## Inference Mode

### Single Inference

Run a single inference and display results:

```bash
gpux run sentiment-analysis --input '{"text": "I love this product!"}'
```

**Output**:
```json
{
  "sentiment": [0.1, 0.9]
}
```

### Batch Inference

For batch inference, pass arrays in input:

```bash
gpux run sentiment --input '{
  "text": ["I love this!", "This is terrible", "Pretty good"]
}'
```

---

## Benchmark Mode

### Basic Benchmark

Run performance benchmark with default settings (100 runs, 10 warmup):

```bash
gpux run sentiment --input '{"text": "Test"}' --benchmark
```

**Output**:

| Metric | Value |
|--------|-------|
| Mean Time | 0.42 ms |
| Min Time | 0.38 ms |
| Max Time | 1.25 ms |
| Std Dev | 0.08 ms |
| Throughput Fps | 2380.9 |

### Custom Benchmark

Specify number of runs and warmup iterations:

```bash
gpux run sentiment \
  --input '{"text": "Test"}' \
  --benchmark \
  --runs 1000 \
  --warmup 50
```

### Save Benchmark Results

Save benchmark metrics to a file:

```bash
gpux run sentiment \
  --input '{"text": "Test"}' \
  --benchmark \
  --runs 1000 \
  --output benchmark.json
```

**benchmark.json**:
```json
{
  "mean_time": 0.42,
  "min_time": 0.38,
  "max_time": 1.25,
  "std_dev": 0.08,
  "throughput_fps": 2380.9,
  "num_runs": 1000,
  "warmup_runs": 50
}
```

---

## Examples

### Sentiment Analysis

```bash
gpux run sentiment-analysis --input '{"text": "I love GPUX!"}'
```

**Output**:
```json
{
  "sentiment": [0.05, 0.95],
  "labels": ["negative", "positive"]
}
```

### Image Classification

```bash
gpux run image-classifier --input '{
  "image": [/* pixel values */]
}'
```

### From File

```bash
gpux run sentiment --file input.json --output result.json
```

### With Specific Provider

```bash
gpux run sentiment \
  --input '{"text": "Test"}' \
  --provider cuda
```

### Benchmark on Apple Silicon

```bash
gpux run sentiment \
  --input '{"text": "Performance test"}' \
  --provider coreml \
  --benchmark \
  --runs 1000
```

---

## Error Handling

### Model Not Found

```bash
Error: Model 'sentiment-analysis' not found
```

**Solution**: Ensure the model exists and `gpux.yml` is configured correctly.

### No Input Data Provided

```bash
Error: No input data provided
```

**Solution**: Provide input using `--input` or `--file`.

### Invalid JSON

```bash
Error parsing input JSON: Expecting property name enclosed in double quotes
```

**Solution**: Ensure your JSON is valid:
```bash
# ❌ Wrong (single quotes)
gpux run sentiment --input "{'text': 'test'}"

# ✅ Correct (double quotes)
gpux run sentiment --input '{"text": "test"}'
```

### Missing Input Fields

```bash
Run failed: Missing required input: 'text'
```

**Solution**: Provide all required inputs as specified in `gpux.yml`.

---

## Best Practices

!!! tip "Use File Input for Large Data"
    For large inputs, use file input instead of command-line JSON:
    ```bash
    gpux run model --file input.json
    ```

!!! tip "Benchmark Before Production"
    Always benchmark your model to understand performance:
    ```bash
    gpux run model --input @test.json --benchmark --runs 1000
    ```

!!! tip "Save Benchmark Results"
    Save benchmark results for performance tracking:
    ```bash
    gpux run model --benchmark --output metrics.json
    ```

!!! warning "Warmup is Important"
    Always include warmup runs for accurate benchmarks. The default is 10, but increase for more stable results:
    ```bash
    gpux run model --benchmark --warmup 50
    ```

!!! tip "Use Verbose Mode for Debugging"
    Enable verbose output to see detailed execution logs:
    ```bash
    gpux run model --input @test.json --verbose
    ```

---

## Performance Tips

1. **Provider Selection**: Use the fastest provider for your hardware:
   - NVIDIA: `--provider tensorrt` or `--provider cuda`
   - Apple Silicon: `--provider coreml`
   - AMD: `--provider rocm`
   - CPU: `--provider cpu`

2. **Batch Processing**: Process multiple items at once for better throughput

3. **Warmup Runs**: Use adequate warmup (50-100 runs) for stable benchmarks

4. **Repeated Benchmarks**: Run benchmarks multiple times and average results

---

## Related Commands

- [`gpux build`](build.md) - Build and validate models
- [`gpux serve`](serve.md) - Start HTTP server for inference
- [`gpux inspect`](inspect.md) - Inspect model details

---

## See Also

- [Running Inference Tutorial](../../tutorial/running-inference.md)
- [Benchmarking Guide](../../tutorial/benchmarking.md)
- [Batch Inference](../../guide/batch-inference.md)
- [Performance Optimization](../../advanced/optimization.md)
