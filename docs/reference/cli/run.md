# `gpux run`

Run inference on models from registries or local projects.

---

## Overview

The `gpux run` command loads a model and runs inference on provided input data. It supports both registry models (pulled from Hugging Face) and local models with `gpux.yml` configuration.

```bash
gpux run MODEL_NAME [OPTIONS]
```

---

## Arguments

### `MODEL_NAME` *(required)*

Name of the model to run. Can be:

- **Registry model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Local model**: `sentiment-analysis` (requires `gpux.yml`)
- **Model path**: `./models/bert` or `/path/to/model`

**Examples**:
```bash
# Registry models
gpux run distilbert-base-uncased-finetuned-sst-2-english
gpux run facebook/opt-125m
gpux run sentence-transformers/all-MiniLM-L6-v2

# Local models
gpux run sentiment-analysis
gpux run image-classifier
gpux run ./models/bert
```

The command searches for models in:
1. Registry cache (`~/.gpux/models/`)
2. Current directory (if `gpux.yml` exists)
3. Directory specified by model name
4. `.gpux/` directory (for built models)

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

### Registry Models

Registry models typically use standardized input formats:

#### Text Classification
```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love GPUX!"}'
```

#### Text Generation
```bash
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'
```

#### Embeddings
```bash
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'
```

#### Question Answering
```bash
gpux run distilbert-base-cased-distilled-squad --input '{"question": "What is AI?", "context": "AI is artificial intelligence"}'
```

### Local Models

Local models use custom input formats defined in `gpux.yml`:

#### JSON String
```bash
gpux run sentiment --input '{"text": "I love GPUX!"}'
```

#### JSON File
```bash
gpux run sentiment --file input.json
```

**input.json**:
```json
{
  "text": "I love GPUX!"
}
```

#### File Path with @ Prefix
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

### Registry Models

#### Sentiment Analysis
```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love GPUX!"}'
```

**Output**:
```json
{
  "logits": [[-3.2, 3.8]],
  "predicted_class": "POSITIVE"
}
```

#### Text Generation
```bash
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'
```

#### Embeddings
```bash
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'
```

#### Question Answering
```bash
gpux run distilbert-base-cased-distilled-squad --input '{"question": "What is AI?", "context": "AI is artificial intelligence"}'
```

### Local Models

#### Sentiment Analysis
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

#### Image Classification
```bash
gpux run image-classifier --input '{
  "image": [/* pixel values */]
}'
```

### File Input

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english --file input.json --output result.json
```

### With Specific Provider

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "Test"}' \
  --provider cuda
```

### Benchmark on Apple Silicon

```bash
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "Performance test"}' \
  --provider coreml \
  --benchmark \
  --runs 1000
```

---

## Error Handling

### Model Not Found

#### Registry Model
```bash
Error: Model 'invalid-model-name' not found in registry
```

**Solution**:
- Check model name spelling
- Verify model exists on Hugging Face Hub
- Try pulling the model first: `gpux pull model-name`

#### Local Model
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
Run failed: Missing required input: 'inputs'
```

**Solution**: Provide all required inputs as specified in model configuration.

### Registry Model Not Cached

```bash
Error: Model 'distilbert-base-uncased-finetuned-sst-2-english' not found in cache
```

**Solution**: Pull the model first:
```bash
gpux pull distilbert-base-uncased-finetuned-sst-2-english
```

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

- [`gpux pull`](pull.md) - Pull models from registries
- [`gpux build`](build.md) - Build and validate local models
- [`gpux serve`](serve.md) - Start HTTP server for inference
- [`gpux inspect`](inspect.md) - Inspect model details

---

## See Also

- [Running Inference Tutorial](../../tutorial/running-inference.md)
- [Pulling Models Tutorial](../../tutorial/pulling-models.md)
- [Working with Registries](../../guide/registries.md)
- [Benchmarking Guide](../../tutorial/benchmarking.md)
- [Batch Inference](../../guide/batch-inference.md)
- [Performance Optimization](../../advanced/optimization.md)
