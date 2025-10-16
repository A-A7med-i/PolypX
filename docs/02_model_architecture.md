# Model Architecture

<div align="center">
  <img src="https://i.postimg.cc/HLS8TMDf/model.png"
       alt="PolypFiLMNet Architecture"
       />
</div>

---

## Introduction

The **PolypFiLMNet** is a deep learning model designed for **colonoscopy polyp segmentation**.
It extends the **DeepLabV3+** architecture by integrating **Feature-wise Linear Modulation (FiLM)**,
allowing the network to adapt feature representations based on **sequence-specific context**.

---

## Key Components

### 1. FiLM Generator

- **Purpose**: Generates scaling (**gamma**) and shifting (**beta**) parameters based on a given *sequence ID*.
- **Process**:
  1. A sequence ID is embedded into a vector space.
  2. Two fully connected layers transform this embedding into gamma and beta values.
  3. The FiLM parameters are applied to feature maps from the encoder.
- **Benefit**: Enables the model to adapt predictions for different image sequences or patients.

---

### 2. PolypFiLMNet

- **Backbone**: Pre-trained **DeepLabV3+** with a ResNet-101 encoder.
- **FiLM Integration**: FiLM modulation is applied at an intermediate encoder layer (defined by `FILM_LAYER`).
- **Decoder & Output**:
  - Features are decoded by DeepLab’s decoder.
  - A segmentation head outputs binary masks for polyps.

---

## Forward Pass Flow

1. Input image `x` is passed through the DeepLabV3+ encoder.
2. Sequence ID `seq_id` is processed by the **FiLM Generator** to produce `gamma` and `beta`.
3. Feature maps at layer `FILM_LAYER` are modulated by FiLM.
4. Modulated features are decoded into a **segmentation mask**.

---

## Code Snippet

```python
from src.model.model import PolypFiLMNet
import torch

# Example usage
model = PolypFiLMNet(num_sequences=3, emb_dim=128)

# Dummy input (batch of 2 RGB images, 256x256)
images = torch.randn(2, 3, 256, 256)

# Dummy sequence IDs
seq_ids = torch.tensor([1, 2])

# Forward pass
outputs = model(images, seq_ids)

print(outputs.shape)  # -> [2, 1, 256, 256]
```

## Advantages of PolypFiLMNet

**Context-Aware Segmentation:**  Adapts predictions to different patient/image sequences.

**Strong Backbone:** Builds on the state-of-the-art DeepLabV3+ with ResNet-101.

**Modular Design:** FiLM is integrated in a plug-and-play manner, making the architecture flexible.

**Improved Accuracy:** FiLM modulation helps refine feature learning for challenging cases

## Next Sections

- [Loss](03_loss.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
