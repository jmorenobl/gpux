# Examples

Real-world examples of using GPUX for different ML tasks.

---

## 🎯 Overview

Learn by example! Each guide includes complete working code, configuration, and explanations.

---

## 📚 Available Examples

### [Sentiment Analysis](sentiment-analysis.md)
**BERT-based text classification**

Classify text sentiment using a fine-tuned BERT model.

- ✅ Text preprocessing and tokenization
- ✅ Binary classification (positive/negative)
- ✅ Complete end-to-end example
- ⏱️ **Time:** 15 minutes

---

### [Image Classification](image-classification.md)
**ResNet-50 for ImageNet**

Classify images into 1000 ImageNet categories.

- ✅ Image preprocessing (resize, normalize)
- ✅ Top-K predictions
- ✅ Batch processing
- ⏱️ **Time:** 20 minutes

---

### [Object Detection](object-detection.md)
**YOLOv8 real-time detection**

Detect objects in images with bounding boxes.

- ✅ YOLO model setup
- ✅ Bounding box predictions
- ✅ NMS post-processing
- ⏱️ **Time:** 25 minutes

---

### [LLM Inference](llm-inference.md)
**Small language model serving**

Run text generation with a small LLM.

- ✅ Tokenization and decoding
- ✅ Text generation
- ✅ Streaming responses
- ⏱️ **Time:** 30 minutes

---

### [Speech Recognition](speech-recognition.md)
**Whisper audio transcription**

Transcribe speech to text using OpenAI Whisper.

- ✅ Audio preprocessing
- ✅ Multi-language support
- ✅ Timestamp generation
- ⏱️ **Time:** 25 minutes

---

### [Embedding Generation](embedding-generation.md)
**Sentence transformers**

Generate vector embeddings for semantic search.

- ✅ Text embeddings
- ✅ Similarity search
- ✅ Batch processing
- ⏱️ **Time:** 20 minutes

---

### [Multi-Modal Models](multi-modal.md)
**CLIP image-text matching**

Match images with text descriptions using CLIP.

- ✅ Image and text encoding
- ✅ Similarity scoring
- ✅ Zero-shot classification
- ⏱️ **Time:** 30 minutes

---

## 🚀 Getting Started

### Prerequisites

- GPUX installed (`uv add gpux`)
- Python 3.11+
- Basic understanding of the [Tutorial](../tutorial/index.md)

### Example Structure

Each example includes:

1. **Overview** - What you'll build
2. **Model Preparation** - Converting/downloading the model
3. **Configuration** - Complete `gpux.yml`
4. **Running** - Step-by-step execution
5. **Results** - Expected output
6. **Production** - Deployment considerations

---

## 📖 How to Use Examples

### Follow Along

Each example is self-contained. Pick one and follow step-by-step.

### Adapt for Your Use Case

Modify examples to fit your specific needs.

### Production Deployment

Examples include production deployment tips.

---

## 💡 Tips

!!! tip "Start Simple"
    Begin with **Sentiment Analysis** - it's the easiest example.

!!! tip "GPU Recommended"
    While examples work on CPU, GPU provides much better performance.

!!! tip "Download Models First"
    Large models may take time to download. Plan accordingly.

---

## 🆘 Need Help?

- 📖 [Tutorial](../tutorial/index.md) - Basic concepts
- 📚 [User Guide](../guide/index.md) - In-depth documentation
- 💬 [Discord](https://discord.gg/gpux) - Community support
- 🐛 [Issues](https://github.com/gpux/gpux-runtime/issues) - Report problems
