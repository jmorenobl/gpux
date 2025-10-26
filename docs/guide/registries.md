# Working with Model Registries

Complete guide to pulling, managing, and using models from Hugging Face and other registries.

---

## 🎯 What You'll Learn

- ✅ Understanding model registries
- ✅ Pulling models from Hugging Face Hub
- ✅ Model caching and version management
- ✅ Working with different model types
- ✅ Authentication and private models
- ✅ Troubleshooting registry issues

---

## 🏪 What are Model Registries?

Model registries are centralized repositories where ML models are stored, versioned, and shared. They provide:

- **Centralized Storage**: Models stored in one place
- **Version Control**: Track model versions and changes
- **Metadata**: Model descriptions, tags, and usage info
- **Access Control**: Public and private model access
- **Integration**: Easy integration with ML frameworks

### Supported Registries

GPUX currently supports:

| Registry | Status | Description |
|----------|--------|-------------|
| **Hugging Face Hub** | ✅ Full Support | 500k+ models, text, vision, audio |
| **ONNX Model Zoo** | 🚧 Planned | Pre-optimized ONNX models |
| **TensorFlow Hub** | 🚧 Planned | TensorFlow models |
| **PyTorch Hub** | 🚧 Planned | PyTorch models |

---

## 🤗 Hugging Face Hub

### Overview

Hugging Face Hub is the largest model registry with over 500,000 models covering:

- **Text Models**: Classification, generation, translation, QA
- **Vision Models**: Image classification, object detection, segmentation
- **Audio Models**: Speech recognition, synthesis, music generation
- **Multimodal Models**: Text + image, video understanding

### Basic Usage

```bash
# Pull a model
gpux pull distilbert-base-uncased-finetuned-sst-2-english

# Specify registry explicitly
gpux pull huggingface:microsoft/DialoGPT-medium

# Use short alias
gpux pull hf:microsoft/DialoGPT-medium
```

### Model Types

#### Text Classification

```bash
# Sentiment analysis
gpux pull distilbert-base-uncased-finetuned-sst-2-english
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love this!"}'

# Topic classification
gpux pull facebook/bart-large-mnli
gpux run facebook/bart-large-mnli --input '{"inputs": "I love pizza", "candidate_labels": ["food", "travel", "sports"]}'
```

#### Text Generation

```bash
# GPT-style models
gpux pull facebook/opt-125m
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'

# Dialog models
gpux pull microsoft/DialoGPT-medium
gpux run microsoft/DialoGPT-medium --input '{"inputs": "Hello, how are you?"}'
```

#### Embeddings

```bash
# General purpose embeddings
gpux pull sentence-transformers/all-MiniLM-L6-v2
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'

# Multilingual embeddings
gpux pull sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
gpux run sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --input '{"inputs": "Bonjour le monde"}'
```

#### Question Answering

```bash
# SQuAD models
gpux pull distilbert-base-cased-distilled-squad
gpux run distilbert-base-cased-distilled-squad --input '{"question": "What is AI?", "context": "AI is artificial intelligence"}'
```

---

## 💾 Model Caching

### Cache Location

Models are cached locally for fast access:

- **macOS/Linux**: `~/.gpux/models/`
- **Windows**: `%USERPROFILE%\.gpux\models\`

### Cache Structure

```
~/.gpux/models/
├── distilbert-base-uncased-finetuned-sst-2-english/
│   ├── model.onnx              # Converted ONNX model
│   ├── gpux.yml               # Auto-generated config
│   ├── tokenizer.json         # Tokenizer files
│   ├── config.json            # Model configuration
│   ├── metadata.json          # GPUX metadata
│   └── cache_info.json        # Cache metadata
├── facebook-opt-125m/
│   ├── model.onnx
│   ├── gpux.yml
│   └── ...
└── sentence-transformers-all-MiniLM-L6-v2/
    ├── model.onnx
    ├── gpux.yml
    └── ...
```

### Cache Management

```bash
# Check cache size
du -sh ~/.gpux/models/

# List cached models
ls ~/.gpux/models/

# Force re-download
gpux pull distilbert-base-uncased-finetuned-sst-2-english --force

# Use custom cache directory
gpux pull microsoft/DialoGPT-medium --cache-dir /path/to/custom/cache
```

---

## 🏷️ Model Versioning

### Revisions and Tags

Hugging Face models support versioning:

```bash
# Pull latest version (default)
gpux pull microsoft/DialoGPT-medium

# Pull specific revision
gpux pull microsoft/DialoGPT-medium --revision v1.0

# Pull specific commit
gpux pull microsoft/DialoGPT-medium --revision abc123def456
```

### Version Information

```bash
# Inspect model to see version info
gpux inspect microsoft/DialoGPT-medium
```

Expected output:
```
╭─ Model Information ─────────────────────────────────────╮
│ Name      │ microsoft/DialoGPT-medium                    │
│ Registry  │ huggingface                                  │
│ Revision  │ main                                         │
│ Size      │ 1.2 GB                                       │
│ Cached    │ ✅ Yes                                       │
╰─────────────────────────────────────────────────────────╯
```

---

## 🔐 Authentication

### Public Models

Most models are public and don't require authentication:

```bash
# Public models work without authentication
gpux pull distilbert-base-uncased-finetuned-sst-2-english
```

### Private Models

For private models, you need authentication:

#### Method 1: Environment Variable

```bash
# Set your Hugging Face token
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Pull private model
gpux pull your-org/private-model
```

#### Method 2: Command Line

```bash
# Pass token directly
gpux pull your-org/private-model --token "hf_your_token_here"
```

#### Method 3: Login (Future)

```bash
# Login to Hugging Face (planned feature)
gpux login huggingface
```

### Getting a Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Copy the token (starts with `hf_`)
4. Set as environment variable or use with `--token`

---

## 🔍 Model Discovery

### Finding Models

#### Hugging Face Hub Website

1. Visit [huggingface.co/models](https://huggingface.co/models)
2. Search for models by task, language, or framework
3. Check model cards for usage examples
4. Note the model ID (e.g., `distilbert-base-uncased-finetuned-sst-2-english`)

#### Popular Model Categories

**Text Classification:**
- `distilbert-base-uncased-finetuned-sst-2-english` - Sentiment analysis
- `cardiffnlp/twitter-roberta-base-sentiment-latest` - Twitter sentiment
- `facebook/bart-large-mnli` - Topic classification

**Text Generation:**
- `facebook/opt-125m` - Small GPT-style model
- `microsoft/DialoGPT-medium` - Dialog generation
- `gpt2` - Original GPT-2

**Embeddings:**
- `sentence-transformers/all-MiniLM-L6-v2` - General purpose
- `sentence-transformers/all-mpnet-base-v2` - Higher quality
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Multilingual

**Question Answering:**
- `distilbert-base-cased-distilled-squad` - SQuAD QA
- `deepset/roberta-base-squad2` - Higher accuracy

### Model Information

```bash
# Get detailed model information
gpux inspect distilbert-base-uncased-finetuned-sst-2-english
```

---

## ⚙️ Advanced Options

### Registry Selection

```bash
# Explicitly specify registry
gpux pull huggingface:microsoft/DialoGPT-medium

# Use short aliases
gpux pull hf:microsoft/DialoGPT-medium
```

### Custom Cache Directory

```bash
# Use custom cache location
gpux pull microsoft/DialoGPT-medium --cache-dir /path/to/custom/cache
```

### Verbose Output

```bash
# Show detailed progress
gpux pull microsoft/DialoGPT-medium --verbose
```

### Force Re-download

```bash
# Force re-download and conversion
gpux pull microsoft/DialoGPT-medium --force
```

---

## 🐛 Troubleshooting

### Common Issues

#### Model Not Found

**Error**: `Model not found: invalid-model-name`

**Solutions**:
- Check model name spelling
- Verify model exists on Hugging Face Hub
- Try with full organization name: `org/model-name`
- Check if model is private (requires authentication)

#### Download Failed

**Error**: `Network error: Failed to download model`

**Solutions**:
- Check internet connection
- Verify Hugging Face Hub is accessible
- Try again with `--force` flag
- Check firewall/proxy settings

#### Conversion Failed

**Error**: `Conversion failed: Unsupported model architecture`

**Solutions**:
- Try a different model
- Check if model supports ONNX conversion
- Use `--verbose` for detailed error information
- Check model compatibility matrix

#### Authentication Failed

**Error**: `Authentication failed: Invalid token`

**Solutions**:
- Verify token is correct
- Check token permissions
- Ensure token starts with `hf_`
- Try regenerating token

#### Memory Issues

**Error**: `Out of memory during conversion`

**Solutions**:
- Try a smaller model
- Close other applications
- Use CPU-only conversion: `--provider cpu`
- Increase system memory

### Debug Mode

```bash
# Enable debug logging
export GPUX_LOG_LEVEL=DEBUG
gpux pull microsoft/DialoGPT-medium --verbose
```

### Cache Issues

```bash
# Clear cache and retry
rm -rf ~/.gpux/models/model-name
gpux pull model-name --force
```

---

## 📊 Performance Tips

### Model Selection

- **Start Small**: Begin with smaller models (< 500MB)
- **Check Compatibility**: Verify ONNX conversion support
- **Consider Use Case**: Choose models optimized for your task

### Caching Strategy

- **Local Cache**: Models are cached locally for fast access
- **Version Pinning**: Pin to specific model versions for production
- **Cache Management**: Monitor cache size and clean up unused models

### Conversion Optimization

- **Batch Size**: Use appropriate batch sizes for your hardware
- **Provider Selection**: Choose optimal execution provider
- **Memory Management**: Monitor memory usage during conversion

---

## 🔮 Future Registries

### Planned Support

**ONNX Model Zoo:**
- Pre-optimized ONNX models
- No conversion required
- Optimized for performance

**TensorFlow Hub:**
- TensorFlow models
- TF.js compatibility
- Google's model repository

**PyTorch Hub:**
- PyTorch models
- TorchScript support
- Facebook's model repository

**MLflow Model Registry:**
- Enterprise model management
- Model versioning and staging
- Integration with MLflow

---

## 💡 Best Practices

### 1. Model Selection

- Choose models appropriate for your task
- Start with smaller models for testing
- Verify ONNX compatibility before pulling

### 2. Version Management

- Pin to specific model versions for production
- Test new versions before upgrading
- Document model versions in your projects

### 3. Authentication

- Use environment variables for tokens
- Never commit tokens to version control
- Rotate tokens regularly

### 4. Cache Management

- Monitor cache size
- Clean up unused models
- Use custom cache directories for different projects

### 5. Error Handling

- Always handle conversion failures gracefully
- Provide fallback options
- Log errors for debugging

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ Understanding model registries and their benefits
    ✅ Pulling models from Hugging Face Hub
    ✅ Working with different model types (classification, generation, embeddings)
    ✅ Model caching and version management
    ✅ Authentication for private models
    ✅ Troubleshooting common registry issues
    ✅ Best practices for registry usage

---

**Previous:** [Models](models.md) | **Next:** [Preprocessing](preprocessing.md)
