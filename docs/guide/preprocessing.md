# Data Preprocessing

Preprocessing pipelines for text, images, and audio.

---

## 🎯 Overview

Learn preprocessing techniques for different data types.

---

## 📝 Text Preprocessing

### Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello world", return_tensors="np")
```

### Configuration

```yaml
preprocessing:
  tokenizer: bert-base-uncased
  max_length: 512
```

---

## 🖼️ Image Preprocessing

### Resize and Normalize

```python
import numpy as np
from PIL import Image

# Load image
img = Image.open("image.jpg")

# Resize
img = img.resize((224, 224))

# Normalize (ImageNet)
img_array = np.array(img) / 255.0
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_normalized = (img_array - mean) / std
```

### Configuration

```yaml
preprocessing:
  resize: [224, 224]
  normalize: imagenet
```

---

## 🎵 Audio Preprocessing

### Resampling

```python
import librosa

audio, sr = librosa.load("audio.wav", sr=16000)
```

---

## 💡 Key Takeaways

!!! success
    ✅ Text tokenization
    ✅ Image preprocessing
    ✅ Audio resampling
    ✅ Configuration options

---

**Previous:** [Inputs & Outputs](inputs-outputs.md) | **Next:** [Batch Inference →](batch-inference.md)
