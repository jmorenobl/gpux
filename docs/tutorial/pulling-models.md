# Pulling Models from Registries

Learn how to pull and use models from Hugging Face and other registries with GPUX.

---

## 🎯 What You'll Learn

- ✅ Pulling models from Hugging Face Hub
- ✅ Understanding model caching and versioning
- ✅ Working with different model types
- ✅ Troubleshooting common issues

---

## 🚀 Basic Usage

### Pull a Model

```bash
# Pull a sentiment analysis model
gpux pull distilbert-base-uncased-finetuned-sst-2-english
```

### Specify Registry

```bash
# Explicitly specify Hugging Face registry
gpux pull huggingface:microsoft/DialoGPT-medium

# Use short alias
gpux pull hf:microsoft/DialoGPT-medium
```

### Pull Specific Version

```bash
# Pull a specific revision/tag
gpux pull microsoft/DialoGPT-medium --revision v1.0
```

---

## 📦 Model Types

### Text Classification

```bash
# Sentiment analysis
gpux pull distilbert-base-uncased-finetuned-sst-2-english

# Run inference
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love this!"}'
```

### Text Generation

```bash
# GPT-style model
gpux pull facebook/opt-125m

# Run inference
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'
```

### Embeddings

```bash
# Sentence embeddings
gpux pull sentence-transformers/all-MiniLM-L6-v2

# Run inference
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'
```

### Question Answering

```bash
# QA model
gpux pull distilbert-base-cased-distilled-squad

# Run inference
gpux run distilbert-base-cased-distilled-squad --input '{"question": "What is AI?", "context": "AI is artificial intelligence"}'
```

---

## 💾 Model Caching

### Cache Location

Models are cached in:
- **macOS/Linux**: `~/.gpux/models/`
- **Windows**: `%USERPROFILE%\.gpux\models\`

### Cache Structure

```
~/.gpux/models/
├── distilbert-base-uncased-finetuned-sst-2-english/
│   ├── model.onnx
│   ├── gpux.yml
│   ├── tokenizer.json
│   └── config.json
└── facebook-opt-125m/
    ├── model.onnx
    ├── gpux.yml
    └── ...
```

### Force Re-download

```bash
# Force re-download and conversion
gpux pull distilbert-base-uncased-finetuned-sst-2-english --force
```

---

## 🔍 Model Information

### Inspect Pulled Model

```bash
# Get detailed information
gpux inspect distilbert-base-uncased-finetuned-sst-2-english
```

Expected output:
```
╭─ Model Information ─────────────────────────────────────╮
│ Name      │ distilbert-base-uncased-finetuned-sst-2-english │
│ Registry  │ huggingface                                     │
│ Size      │ 268 MB                                           │
│ Format    │ onnx                                            │
│ Cached    │ ✅ Yes                                          │
╰─────────────────────────────────────────────────────────╯

╭─ Input Specifications ──────────────────────────────────╮
│ Name   │ Type    │ Shape     │ Required │
│ inputs │ string  │ variable  │ ✅       │
╰─────────────────────────────────────────────────────────╯

╭─ Output Specifications ─────────────────────────────────╮
│ Name   │ Type    │ Shape    │
│ logits │ float32 │ [1, 2]   │
╰─────────────────────────────────────────────────────────╯
```

---

## ⚙️ Advanced Options

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

### Authentication

For private models, set your Hugging Face token:

```bash
# Set environment variable
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Or use --token parameter
gpux pull your-org/private-model --token "your_token_here"
```

---

## 🐛 Troubleshooting

### Model Not Found

**Error**: `Model not found: invalid-model-name`

**Solutions**:
- Check model name spelling
- Verify model exists on Hugging Face Hub
- Try with full organization name: `org/model-name`

### Download Failed

**Error**: `Network error: Failed to download model`

**Solutions**:
- Check internet connection
- Verify Hugging Face Hub is accessible
- Try again with `--force` flag

### Conversion Failed

**Error**: `Conversion failed: Unsupported model architecture`

**Solutions**:
- Try a different model
- Check if model supports ONNX conversion
- Use `--verbose` for detailed error information

### Memory Issues

**Error**: `Out of memory during conversion`

**Solutions**:
- Try a smaller model
- Close other applications
- Use CPU-only conversion: `--provider cpu`

---

## 📚 Popular Models

### Text Classification

```bash
# Sentiment analysis
gpux pull distilbert-base-uncased-finetuned-sst-2-english
gpux pull cardiffnlp/twitter-roberta-base-sentiment-latest

# Topic classification
gpux pull facebook/bart-large-mnli
```

### Text Generation

```bash
# Small models
gpux pull facebook/opt-125m
gpux pull microsoft/DialoGPT-small

# Medium models
gpux pull microsoft/DialoGPT-medium
gpux pull facebook/opt-350m
```

### Embeddings

```bash
# General purpose
gpux pull sentence-transformers/all-MiniLM-L6-v2
gpux pull sentence-transformers/all-mpnet-base-v2

# Multilingual
gpux pull sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Question Answering

```bash
# SQuAD models
gpux pull distilbert-base-cased-distilled-squad
gpux pull deepset/roberta-base-squad2
```

---

## 💡 Best Practices

### 1. Start Small

Begin with smaller models to test your setup:

```bash
# Start with lightweight models
gpux pull distilbert-base-uncased-finetuned-sst-2-english
gpux pull sentence-transformers/all-MiniLM-L6-v2
```

### 2. Check Model Size

Before pulling large models, check their size:

```bash
# Check model info before downloading
curl -s "https://huggingface.co/api/models/microsoft/DialoGPT-medium" | jq '.safetensors.total'
```

### 3. Use Specific Versions

For production, pin to specific model versions:

```bash
# Use specific revision
gpux pull microsoft/DialoGPT-medium --revision v1.0
```

### 4. Monitor Cache Usage

Keep track of your cache size:

```bash
# Check cache size
du -sh ~/.gpux/models/
```

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ How to pull models from Hugging Face Hub
    ✅ Understanding model caching and versioning
    ✅ Working with different model types (classification, generation, embeddings)
    ✅ Troubleshooting common pull issues
    ✅ Best practices for model management

---

**Previous:** [First Steps](first-steps.md) | **Next:** [Running Inference](running-inference.md)
