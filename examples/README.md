# GPUX Examples

This directory contains example GPUX projects demonstrating different types of ML models and use cases.

## Examples

### Sentiment Analysis
- **Model**: BERT-based sentiment analysis
- **Input**: Text strings
- **Output**: Sentiment classification (positive/negative)
- **File**: `sentiment-analysis/gpux.yml`

### Image Classification
- **Model**: ResNet-50 for ImageNet
- **Input**: Images
- **Output**: ImageNet class probabilities
- **File**: `image-classification/gpux.yml`

## Usage

1. **Build a model**:
   ```bash
   cd examples/sentiment-analysis
   gpux build .
   ```

2. **Run inference**:
   ```bash
   gpux run sentiment-analysis --input '{"input_ids": [101, 2023, 2003, 1037, 3231, 102], "attention_mask": [1, 1, 1, 1, 1, 1]}'
   ```

3. **Start a server**:
   ```bash
   gpux serve sentiment-analysis --port 8080
   ```

4. **Inspect model**:
   ```bash
   gpux inspect sentiment-analysis
   ```

## Creating Your Own Example

1. Create a new directory for your model
2. Add your ONNX model file
3. Create a `gpux.yml` with the appropriate configuration
4. Test with `gpux build .` and `gpux run <model-name>`
