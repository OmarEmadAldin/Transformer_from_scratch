# Transformer From Scratch - Arabic to English Translation

## Overview

This project implements the Transformer architecture from scratch using PyTorch for Arabic-English machine translation.

The implementation includes:

- Input Embeddings
- Positional Encoding
- Multi-Head Attention
- Feed Forward Networks
- Layer Normalization
- Residual Connections
- Encoder
- Decoder
- Greedy Decoding

The model is trained on the OPUS-100 Arabic-English dataset.

---

## Project Structure

```text
.
├── model.py        # Transformer implementation
├── train.py        # Training pipeline
├── dataset.py      # Dataset preprocessing
├── config.py       # Configuration file
└── weights/        # Saved checkpoints
```

---

## Dataset

Dataset: OPUS-100 (Arabic-English)

```python
load_dataset("Helsinki-NLP/opus-100", "ar-en")
```

Training split:
- 90% Training
- 10% Validation

---

## Model Architecture

### Encoder
- Multi-Head Self Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization

### Decoder
- Masked Multi-Head Self Attention
- Cross Attention
- Feed Forward Network
- Residual Connections
- Layer Normalization

---

## Training

Loss Function:

```python
nn.CrossEntropyLoss(
    ignore_index=PAD_ID,
    label_smoothing=0.1
)
```

Optimizer:

```python
torch.optim.Adam(
    model.parameters(),
    lr=config["lr"]
)
```

---

## Evaluation Metrics

The model is evaluated using:

- BLEU Score
- Character Error Rate (CER)
- Word Error Rate (WER)

---

## Run Training

```bash
python train.py
```

