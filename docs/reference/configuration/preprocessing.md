# Preprocessing Configuration

Data preprocessing settings in `gpux.yml`.

---

## Overview

The `preprocessing` section defines data preprocessing pipelines.

```yaml
preprocessing:
  tokenizer: string       # Tokenizer name
  max_length: int         # Max tokenization length
  resize: [int, int]      # Image resize dimensions
  normalize: string       # Normalization method
  custom: {}              # Custom preprocessing config
```

---

## Fields

### `tokenizer`

Tokenizer name for text preprocessing.

- **Type**: `string`
- **Required**: No
- **Examples**: HuggingFace tokenizer names

```yaml
preprocessing:
  tokenizer: bert-base-uncased
  tokenizer: gpt2
  tokenizer: distilbert-base-uncased
```

### `max_length`

Maximum sequence length for tokenization.

- **Type**: `integer`
- **Required**: No

```yaml
preprocessing:
  max_length: 128
  max_length: 512
```

### `resize`

Image resize dimensions `[width, height]`.

- **Type**: `list[int, int]`
- **Required**: No

```yaml
preprocessing:
  resize: [224, 224]   # Square resize
  resize: [640, 480]   # Rectangle resize
```

### `normalize`

Normalization method for images.

- **Type**: `string`
- **Required**: No
- **Values**: `imagenet`, `custom`, or specific normalization

```yaml
preprocessing:
  normalize: imagenet  # ImageNet normalization
```

### `custom`

Custom preprocessing configuration.

- **Type**: `dict`
- **Required**: No

```yaml
preprocessing:
  custom:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

---

## Examples

### Text Preprocessing

```yaml
preprocessing:
  tokenizer: bert-base-uncased
  max_length: 128
```

### Image Preprocessing

```yaml
preprocessing:
  resize: [224, 224]
  normalize: imagenet
```

### Custom Preprocessing

```yaml
preprocessing:
  custom:
    mean: [0.5, 0.5, 0.5]
    std: [0.5, 0.5, 0.5]
    resize: [256, 256]
    crop: [224, 224]
```

---

## Complete Examples

### BERT Sentiment Analysis

```yaml
name: sentiment-analysis
model:
  source: ./bert.onnx

inputs:
  - name: input_ids
    type: int64
    shape: [1, 128]
  - name: attention_mask
    type: int64
    shape: [1, 128]

outputs:
  - name: logits
    type: float32
    shape: [1, 2]

preprocessing:
  tokenizer: bert-base-uncased
  max_length: 128
```

### Image Classification

```yaml
name: image-classifier
model:
  source: ./resnet50.onnx

inputs:
  - name: image
    type: float32
    shape: [1, 3, 224, 224]

outputs:
  - name: probabilities
    type: float32
    shape: [1, 1000]

preprocessing:
  resize: [224, 224]
  normalize: imagenet
```

---

## See Also

- [Configuration Schema](schema.md)
- [Preprocessing Guide](../../guide/preprocessing.md)
