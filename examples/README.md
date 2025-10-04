# GPUX Examples

This directory contains example configurations and usage patterns for different types of ML models with GPUX.

## 📁 Available Examples

### 1. Sentiment Analysis (`sentiment-analysis/`)
BERT-based text classification for sentiment analysis.

**Features:**
- Text preprocessing with tokenization
- BERT model inference
- Binary classification (positive/negative)

**Usage:**
```bash
cd examples/sentiment-analysis
gpux build .
gpux run sentiment-analysis --input '{"text": "I love this product!"}'
```

### 2. Image Classification (`image-classification/`)
ResNet-50 model for ImageNet image classification.

**Features:**
- Image preprocessing (resize, normalize)
- ResNet-50 inference
- 1000-class ImageNet classification

**Usage:**
```bash
cd examples/image-classification
gpux build .
gpux run image-classification --file input.json
```

### 3. Object Detection (`object-detection/`)
YOLOv8 model for real-time object detection.

**Features:**
- YOLOv8 inference
- Real-time detection capabilities
- Bounding box predictions

**Usage:**
```bash
cd examples/object-detection
gpux build .
gpux serve object-detection --port 8080
```

### 4. LLM Chat (`llm-chat/`)
Small language model for chat and text generation.

**Features:**
- Language model inference
- Text generation capabilities
- Chat conversation handling

**Usage:**
```bash
cd examples/llm-chat
gpux build .
gpux run llm-chat --input '{"prompt": "Hello, how are you?"}'
```

### 5. Speech Recognition (`speech-recognition/`)
Whisper model for audio transcription.

**Features:**
- Audio preprocessing
- Whisper inference
- Speech-to-text transcription

**Usage:**
```bash
cd examples/speech-recognition
gpux build .
gpux run speech-recognition --file audio.json
```

## 🚀 Getting Started with Examples

### Prerequisites

1. **Install GPUX**
   ```bash
   pip install gpux
   ```

2. **Download Model Files**
   Each example requires a corresponding ONNX model file. You can:
   - Use the provided model files (if available)
   - Convert your own models to ONNX format
   - Download pre-trained models from model hubs

### Running an Example

1. **Navigate to the example directory**
   ```bash
   cd examples/sentiment-analysis
   ```

2. **Build the model**
   ```bash
   gpux build .
   ```

3. **Run inference**
   ```bash
   # Single inference
   gpux run sentiment-analysis --input '{"text": "This is great!"}'

   # From file
   gpux run sentiment-analysis --file input.json

   # Benchmark
   gpux run sentiment-analysis --benchmark --runs 1000
   ```

4. **Start a server (optional)**
   ```bash
   gpux serve sentiment-analysis --port 8080
   ```

## 📝 Input Data Formats

### Text Input (Sentiment Analysis, LLM Chat)
```json
{
  "text": "Your input text here",
  "max_length": 512
}
```

### Image Input (Image Classification, Object Detection)
```json
{
  "image": "base64_encoded_image_data",
  "format": "jpeg"
}
```

### Audio Input (Speech Recognition)
```json
{
  "audio": "base64_encoded_audio_data",
  "sample_rate": 16000,
  "format": "wav"
}
```

## 🔧 Customizing Examples

### Modifying Configuration

Edit the `gpux.yml` file to customize:

- **Model path**: Update the `model.source` field
- **Input/Output specs**: Modify the `inputs` and `outputs` sections
- **Runtime settings**: Adjust GPU memory, batch size, etc.
- **Preprocessing**: Update preprocessing parameters

### Adding New Examples

1. **Create a new directory**
   ```bash
   mkdir examples/my-new-example
   cd examples/my-new-example
   ```

2. **Create gpux.yml**
   ```yaml
   name: my-new-example
   version: 1.0.0
   description: "Description of your example"

   model:
     source: ./my_model.onnx
     format: onnx

   inputs:
     input_name:
       type: float32
       shape: [1, 10]
       required: true
       description: "Input description"

   outputs:
     output_name:
       type: float32
       shape: [1, 2]
       description: "Output description"

   runtime:
     gpu:
       memory: 2GB
       backend: auto
   ```

3. **Add example data**
   Create sample input files for testing.

4. **Test your example**
   ```bash
   gpux build .
   gpux run my-new-example --input '{"data": [1,2,3]}'
   ```

## 🎯 Performance Tips

### GPU Memory Optimization
- Start with smaller models for testing
- Adjust `gpu.memory` based on your GPU
- Use `batch_size: 1` for memory-constrained environments

### Provider Selection
- Use `backend: auto` for automatic selection
- Specify `backend: cuda` for NVIDIA GPUs
- Use `backend: coreml` for Apple Silicon

### Benchmarking
- Use `--warmup` to warm up the model
- Run multiple `--runs` for accurate metrics
- Test with different input sizes

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure the model file exists at the specified path
   - Check file permissions

2. **Out of memory**
   - Reduce `gpu.memory` allocation
   - Use `batch_size: 1`
   - Try CPU fallback

3. **Provider not available**
   - Check GPU drivers
   - Verify ONNX Runtime installation
   - Use `gpux inspect` to see available providers

4. **Input validation errors**
   - Check input data format
   - Verify input shapes match model expectations
   - Ensure required fields are provided

### Getting Help

- Check the [API documentation](../docs/API.md)
- Review the [architecture guide](../docs/ARCHITECTURE.md)
- Open an [issue](https://github.com/gpux/gpux-runtime/issues) for bugs
- Start a [discussion](https://github.com/gpux/gpux-runtime/discussions) for questions

## 📚 Additional Resources

- [GPUX Documentation](../docs/)
- [API Reference](../docs/API.md)
- [Architecture Guide](../docs/ARCHITECTURE.md)
- [Contributing Guide](../docs/CONTRIBUTING.md)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [ONNX Model Zoo](https://github.com/onnx/models)
