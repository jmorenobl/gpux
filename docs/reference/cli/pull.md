# gpux pull

Pull models from registries and convert them to ONNX format.

---

## Synopsis

```bash
gpux pull <model-id> [OPTIONS]
```

---

## Description

The `gpux pull` command downloads models from supported registries (currently Hugging Face Hub) and automatically converts them to ONNX format for use with GPUX. Models are cached locally for fast access.

---

## Arguments

### `<model-id>`

The model identifier. Can be specified in several formats:

- **Simple format**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Registry format**: `huggingface:microsoft/DialoGPT-medium`
- **Short alias**: `hf:microsoft/DialoGPT-medium`

---

## Options

### `--registry`, `-r`

Specify the registry to pull from.

```bash
gpux pull microsoft/DialoGPT-medium --registry huggingface
```

**Default**: `huggingface`

**Supported values**:
- `huggingface` - Hugging Face Hub
- `hf` - Short alias for Hugging Face Hub

### `--revision`, `--rev`

Pull a specific revision or tag of the model.

```bash
gpux pull microsoft/DialoGPT-medium --revision v1.0
gpux pull microsoft/DialoGPT-medium --revision abc123def456
```

**Default**: `main` (latest)

### `--cache-dir`

Specify a custom cache directory.

```bash
gpux pull microsoft/DialoGPT-medium --cache-dir /path/to/custom/cache
```

**Default**: `~/.gpux/models/` (macOS/Linux) or `%USERPROFILE%\.gpux\models\` (Windows)

### `--token`

Authentication token for private models.

```bash
gpux pull your-org/private-model --token "hf_your_token_here"
```

**Note**: You can also set the `HUGGINGFACE_HUB_TOKEN` environment variable.

### `--force`, `-f`

Force re-download and conversion, even if the model is already cached.

```bash
gpux pull microsoft/DialoGPT-medium --force
```

### `--provider`

Specify the execution provider for conversion.

```bash
gpux pull microsoft/DialoGPT-medium --provider cpu
gpux pull microsoft/DialoGPT-medium --provider cuda
```

**Default**: `auto` (automatically select best available)

**Supported values**:
- `auto` - Automatically select best provider
- `cpu` - CPU only
- `cuda` - NVIDIA CUDA
- `coreml` - Apple CoreML
- `rocm` - AMD ROCm
- `directml` - Windows DirectML

### `--verbose`, `-v`

Enable verbose output showing detailed progress.

```bash
gpux pull microsoft/DialoGPT-medium --verbose
```

### `--help`, `-h`

Show help message and exit.

---

## Examples

### Basic Usage

```bash
# Pull a sentiment analysis model
gpux pull distilbert-base-uncased-finetuned-sst-2-english

# Pull a text generation model
gpux pull facebook/opt-125m

# Pull an embedding model
gpux pull sentence-transformers/all-MiniLM-L6-v2
```

### Registry Specification

```bash
# Explicitly specify Hugging Face registry
gpux pull huggingface:microsoft/DialoGPT-medium

# Use short alias
gpux pull hf:microsoft/DialoGPT-medium
```

### Version Control

```bash
# Pull specific revision
gpux pull microsoft/DialoGPT-medium --revision v1.0

# Pull specific commit
gpux pull microsoft/DialoGPT-medium --revision abc123def456
```

### Authentication

```bash
# Pull private model with token
gpux pull your-org/private-model --token "hf_your_token_here"

# Using environment variable
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
gpux pull your-org/private-model
```

### Advanced Options

```bash
# Force re-download
gpux pull microsoft/DialoGPT-medium --force

# Use custom cache directory
gpux pull microsoft/DialoGPT-medium --cache-dir /path/to/cache

# Verbose output
gpux pull microsoft/DialoGPT-medium --verbose

# CPU-only conversion
gpux pull microsoft/DialoGPT-medium --provider cpu
```

---

## Output

### Success Output

```
╭─ Pulling Model ────────────────────────────────────────────────╮
│ Registry: huggingface                                           │
│ Model: microsoft/DialoGPT-medium                                │
│ Revision: main                                                  │
│ Size: 1.2 GB                                                    │
╰─────────────────────────────────────────────────────────────────╯

📥 Downloading model files...
✅ Model downloaded successfully!

🔄 Converting to ONNX...
✅ Conversion completed!

📝 Generating configuration...
✅ Configuration saved to: ~/.gpux/models/microsoft-DialoGPT-medium/gpux.yml

🎉 Model ready! Use: gpux run microsoft/DialoGPT-medium
```

### Verbose Output

```
╭─ Pulling Model ────────────────────────────────────────────────╮
│ Registry: huggingface                                           │
│ Model: microsoft/DialoGPT-medium                                │
│ Revision: main                                                  │
│ Size: 1.2 GB                                                    │
╰─────────────────────────────────────────────────────────────────╯

📥 Downloading model files...
  └─ Downloading config.json... ✅
  └─ Downloading pytorch_model.bin... ✅
  └─ Downloading tokenizer.json... ✅
  └─ Downloading tokenizer_config.json... ✅
✅ Model downloaded successfully!

🔄 Converting to ONNX...
  └─ Loading PyTorch model... ✅
  └─ Exporting to ONNX... ✅
  └─ Validating ONNX model... ✅
✅ Conversion completed!

📝 Generating configuration...
  └─ Analyzing model inputs... ✅
  └─ Analyzing model outputs... ✅
  └─ Generating gpux.yml... ✅
✅ Configuration saved to: ~/.gpux/models/microsoft-DialoGPT-medium/gpux.yml

🎉 Model ready! Use: gpux run microsoft/DialoGPT-medium
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Model not found
- `3` - Network error
- `4` - Authentication error
- `5` - Conversion error

---

## Environment Variables

### `HUGGINGFACE_HUB_TOKEN`

Authentication token for Hugging Face Hub.

```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

### `GPUX_CACHE_DIR`

Default cache directory for models.

```bash
export GPUX_CACHE_DIR="/path/to/custom/cache"
```

### `GPUX_LOG_LEVEL`

Logging level for debugging.

```bash
export GPUX_LOG_LEVEL="DEBUG"
```

---

## Cache Management

### Cache Location

Models are cached in:
- **macOS/Linux**: `~/.gpux/models/`
- **Windows**: `%USERPROFILE%\.gpux\models\`

### Cache Structure

```
~/.gpux/models/
├── microsoft-DialoGPT-medium/
│   ├── model.onnx              # Converted ONNX model
│   ├── gpux.yml               # Auto-generated config
│   ├── tokenizer.json         # Tokenizer files
│   ├── config.json            # Model configuration
│   └── metadata.json          # GPUX metadata
└── distilbert-base-uncased-finetuned-sst-2-english/
    ├── model.onnx
    ├── gpux.yml
    └── ...
```

### Cache Operations

```bash
# Check cache size
du -sh ~/.gpux/models/

# List cached models
ls ~/.gpux/models/

# Clear specific model cache
rm -rf ~/.gpux/models/model-name

# Clear all cache
rm -rf ~/.gpux/models/
```

---

## Troubleshooting

### Common Issues

#### Model Not Found

**Error**: `Model not found: invalid-model-name`

**Solutions**:
- Check model name spelling
- Verify model exists on Hugging Face Hub
- Try with full organization name: `org/model-name`

#### Download Failed

**Error**: `Network error: Failed to download model`

**Solutions**:
- Check internet connection
- Verify Hugging Face Hub is accessible
- Try again with `--force` flag

#### Conversion Failed

**Error**: `Conversion failed: Unsupported model architecture`

**Solutions**:
- Try a different model
- Check if model supports ONNX conversion
- Use `--verbose` for detailed error information

#### Authentication Failed

**Error**: `Authentication failed: Invalid token`

**Solutions**:
- Verify token is correct
- Check token permissions
- Ensure token starts with `hf_`

#### Memory Issues

**Error**: `Out of memory during conversion`

**Solutions**:
- Try a smaller model
- Close other applications
- Use CPU-only conversion: `--provider cpu`

---

## Related Commands

- [`gpux run`](run.md) - Run inference on pulled models
- [`gpux inspect`](inspect.md) - Inspect model information
- [`gpux serve`](serve.md) - Serve models via HTTP API

---

## See Also

- [Pulling Models Tutorial](../../tutorial/pulling-models.md)
- [Working with Registries](../../guide/registries.md)
- [Hugging Face Hub](https://huggingface.co/models)
