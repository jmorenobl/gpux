# Hugging Face Models Examples

Real-world examples of using popular Hugging Face models with GPUX.

---

## 🎯 What You'll Learn

- ✅ Pulling and running popular Hugging Face models
- ✅ Different model types and their use cases
- ✅ Input/output formats for each model type
- ✅ Performance characteristics and optimization tips
- ✅ Common troubleshooting scenarios

---

## 📝 Text Classification

### Sentiment Analysis

#### DistilBERT Sentiment Analysis

```bash
# Pull the model
gpux pull distilbert-base-uncased-finetuned-sst-2-english

# Run inference
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "I love this product!"}'
```

**Expected Output**:
```json
{
  "logits": [[-3.2, 3.8]],
  "predicted_class": "POSITIVE"
}
```

**Batch Processing**:
```bash
# Create batch input file
cat > batch_sentiment.json << EOF
[
  {"inputs": "I love this product!"},
  {"inputs": "This is terrible."},
  {"inputs": "It's okay, nothing special."}
]
EOF

# Run batch inference
gpux run distilbert-base-uncased-finetuned-sst-2-english --file batch_sentiment.json
```

#### Twitter Sentiment Analysis

```bash
# Pull Twitter-specific model
gpux pull cardiffnlp/twitter-roberta-base-sentiment-latest

# Run inference
gpux run cardiffnlp/twitter-roberta-base-sentiment-latest --input '{"inputs": "Just had the best coffee ever! ☕️"}'
```

### Topic Classification

#### BART MNLI (Zero-shot Classification)

```bash
# Pull BART MNLI model
gpux pull facebook/bart-large-mnli

# Run zero-shot classification
gpux run facebook/bart-large-mnli --input '{
  "inputs": "I love pizza",
  "candidate_labels": ["food", "travel", "sports"]
}'
```

**Expected Output**:
```json
{
  "sequence": "I love pizza",
  "labels": ["food", "travel", "sports"],
  "scores": [0.95, 0.03, 0.02]
}
```

---

## 🗣️ Text Generation

### GPT-style Models

#### OPT-125M (Small GPT Model)

```bash
# Pull OPT model
gpux pull facebook/opt-125m

# Run text generation
gpux run facebook/opt-125m --input '{"inputs": "The future of AI is"}'
```

**Expected Output**:
```json
{
  "generated_text": "The future of AI is bright and full of possibilities. As technology continues to advance..."
}
```

#### DialoGPT (Dialog Generation)

```bash
# Pull DialoGPT model
gpux pull microsoft/DialoGPT-medium

# Run dialog generation
gpux run microsoft/DialoGPT-medium --input '{"inputs": "Hello, how are you?"}'
```

**Expected Output**:
```json
{
  "generated_text": "Hello! I'm doing well, thank you for asking. How are you doing today?"
}
```

### Advanced Text Generation

#### With Parameters

```bash
# Text generation with parameters
gpux run facebook/opt-125m --input '{
  "inputs": "The future of AI is",
  "max_length": 50,
  "temperature": 0.7,
  "do_sample": true
}'
```

---

## 🔍 Question Answering

### SQuAD Models

#### DistilBERT SQuAD

```bash
# Pull DistilBERT SQuAD model
gpux pull distilbert-base-cased-distilled-squad

# Run question answering
gpux run distilbert-base-cased-distilled-squad --input '{
  "question": "What is artificial intelligence?",
  "context": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."
}'
```

**Expected Output**:
```json
{
  "answer": "intelligence demonstrated by machines",
  "score": 0.95,
  "start": 0,
  "end": 42
}
```

#### RoBERTa SQuAD (Higher Accuracy)

```bash
# Pull RoBERTa SQuAD model
gpux pull deepset/roberta-base-squad2

# Run question answering
gpux run deepset/roberta-base-squad2 --input '{
  "question": "When was GPUX created?",
  "context": "GPUX is a Docker-like runtime for ML inference that was created in 2025 to solve GPU compatibility issues."
}'
```

---

## 🧮 Embeddings

### Sentence Embeddings

#### All-MiniLM-L6-v2 (General Purpose)

```bash
# Pull embedding model
gpux pull sentence-transformers/all-MiniLM-L6-v2

# Generate embeddings
gpux run sentence-transformers/all-MiniLM-L6-v2 --input '{"inputs": "Hello world"}'
```

**Expected Output**:
```json
{
  "embeddings": [[0.1, -0.2, 0.3, ...]]  // 384-dimensional vector
}
```

#### All-mpnet-base-v2 (Higher Quality)

```bash
# Pull higher quality embedding model
gpux pull sentence-transformers/all-mpnet-base-v2

# Generate embeddings
gpux run sentence-transformers/all-mpnet-base-v2 --input '{"inputs": "Hello world"}'
```

**Expected Output**:
```json
{
  "embeddings": [[0.1, -0.2, 0.3, ...]]  // 768-dimensional vector
}
```

### Multilingual Embeddings

#### Multilingual MiniLM

```bash
# Pull multilingual model
gpux pull sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Generate embeddings in different languages
gpux run sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --input '{"inputs": "Bonjour le monde"}'
gpux run sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --input '{"inputs": "Hola mundo"}'
```

---

## 🌐 Serving Models

### HTTP API Server

#### Start Server

```bash
# Start server with sentiment model
gpux serve distilbert-base-uncased-finetuned-sst-2-english --port 8080
```

#### Test API

```bash
# Test sentiment analysis API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": "I love this product!"}'
```

**Expected Response**:
```json
{
  "logits": [[-3.2, 3.8]],
  "predicted_class": "POSITIVE"
}
```

#### Batch API Requests

```bash
# Test batch processing
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"inputs": "I love this!"},
    {"inputs": "This is terrible."},
    {"inputs": "It'\''s okay."}
  ]'
```

### Health Check

```bash
# Check server health
curl http://localhost:8080/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "uptime": "00:05:23"
}
```

---

## 🔍 Model Inspection

### Inspect Model Details

```bash
# Inspect sentiment model
gpux inspect distilbert-base-uncased-finetuned-sst-2-english
```

**Expected Output**:
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

╭─ Runtime Information ───────────────────────────────────╮
│ Provider  │ CoreMLExecutionProvider                     │
│ Backend   │ auto                                        │
│ GPU Memory│ 2GB                                         │
╰─────────────────────────────────────────────────────────╯
```

### Compare Models

```bash
# Inspect different models
gpux inspect distilbert-base-uncased-finetuned-sst-2-english
gpux inspect facebook/opt-125m
gpux inspect sentence-transformers/all-MiniLM-L6-v2
```

---

## ⚡ Performance Optimization

### Provider Selection

#### Apple Silicon (M1/M2/M3)

```bash
# Use CoreML for best performance
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --provider coreml
```

#### NVIDIA GPUs

```bash
# Use CUDA for NVIDIA GPUs
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --provider cuda

# Use TensorRT for maximum performance
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --provider tensorrt
```

#### AMD GPUs

```bash
# Use ROCm for AMD GPUs
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --provider rocm
```

### Benchmarking

#### Basic Benchmark

```bash
# Run benchmark
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --benchmark
```

**Expected Output**:
```
╭─ Benchmark Results ────────────────────────────────────╮
│ Metric         │ Value                                 │
│ Mean Time      │ 0.42 ms                               │
│ Min Time       │ 0.38 ms                               │
│ Max Time       │ 1.25 ms                               │
│ Std Dev        │ 0.08 ms                               │
│ Throughput Fps │ 2380.9                                │
╰─────────────────────────────────────────────────────────╯
```

#### Custom Benchmark

```bash
# Custom benchmark with more runs
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --benchmark \
  --runs 1000 \
  --warmup 50
```

#### Save Benchmark Results

```bash
# Save benchmark to file
gpux run distilbert-base-uncased-finetuned-sst-2-english \
  --input '{"inputs": "test"}' \
  --benchmark \
  --runs 1000 \
  --output benchmark_results.json
```

---

## 🐛 Troubleshooting

### Common Issues

#### Model Not Found

**Error**: `Model not found: invalid-model-name`

**Solution**:
```bash
# Check model name on Hugging Face Hub
# Pull the correct model
gpux pull distilbert-base-uncased-finetuned-sst-2-english
```

#### Conversion Failed

**Error**: `Conversion failed: Unsupported model architecture`

**Solution**:
```bash
# Try a different model
gpux pull facebook/opt-125m

# Use verbose mode for details
gpux pull microsoft/DialoGPT-medium --verbose
```

#### Memory Issues

**Error**: `Out of memory during conversion`

**Solution**:
```bash
# Use CPU-only conversion
gpux pull microsoft/DialoGPT-medium --provider cpu

# Try a smaller model
gpux pull facebook/opt-125m
```

#### Input Format Errors

**Error**: `Missing required input: 'inputs'`

**Solution**:
```bash
# Check correct input format
gpux inspect distilbert-base-uncased-finetuned-sst-2-english

# Use correct input format
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "test"}'
```

### Debug Mode

```bash
# Enable debug logging
export GPUX_LOG_LEVEL=DEBUG
gpux run distilbert-base-uncased-finetuned-sst-2-english --input '{"inputs": "test"}' --verbose
```

---

## 📊 Performance Comparison

### Model Size vs Performance

| Model | Size | Type | Use Case | Performance |
|-------|------|------|----------|-------------|
| `distilbert-base-uncased-finetuned-sst-2-english` | 268 MB | Classification | Sentiment | Fast |
| `facebook/opt-125m` | 500 MB | Generation | Text | Medium |
| `microsoft/DialoGPT-medium` | 1.2 GB | Generation | Dialog | Slower |
| `sentence-transformers/all-MiniLM-L6-v2` | 90 MB | Embeddings | Similarity | Very Fast |
| `facebook/bart-large-mnli` | 1.6 GB | Classification | Zero-shot | Slower |

### Provider Performance (Apple Silicon)

| Provider | Speed | Memory | Compatibility |
|----------|-------|--------|---------------|
| CoreML | Fastest | Low | Apple Silicon only |
| CPU | Slowest | Lowest | Universal |
| Auto | Optimal | Medium | Universal |

---

## 💡 Best Practices

### 1. Model Selection

- **Start Small**: Begin with smaller models for testing
- **Check Compatibility**: Verify ONNX conversion support
- **Consider Use Case**: Choose models optimized for your task

### 2. Input Formatting

- **Use Standard Formats**: Follow Hugging Face conventions
- **Validate Inputs**: Check input format before running
- **Batch Processing**: Use arrays for multiple inputs

### 3. Performance Optimization

- **Provider Selection**: Use optimal provider for your hardware
- **Benchmarking**: Always benchmark before production
- **Caching**: Models are cached locally for fast access

### 4. Error Handling

- **Graceful Degradation**: Handle conversion failures
- **Fallback Options**: Have alternative models ready
- **Logging**: Enable verbose logging for debugging

---

## 🔗 Related Resources

- [Pulling Models Tutorial](../tutorial/pulling-models.md)
- [Working with Registries](../guide/registries.md)
- [Running Inference Tutorial](../tutorial/running-inference.md)
- [Hugging Face Hub](https://huggingface.co/models)
- [Hugging Face Documentation](https://huggingface.co/docs)

---

## 💡 Key Takeaways

!!! success "What You Learned"
    ✅ How to pull and run popular Hugging Face models
    ✅ Different model types and their specific use cases
    ✅ Input/output formats for each model type
    ✅ Performance optimization techniques
    ✅ Troubleshooting common issues
    ✅ Best practices for production usage

---

**Previous:** [Multi-modal](multi-modal.md) | **Next:** [Speech Recognition](speech-recognition.md)
