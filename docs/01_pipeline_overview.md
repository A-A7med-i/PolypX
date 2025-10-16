# Pipeline Overview

## Introduction

The **PolypX Segmentation Pipeline** offers an end-to-end, modular workflow for medical image segmentation. It covers every stage—from data loading and preprocessing to model training, evaluation, and inference—using reusable, well-structured components.

---

## Pipeline Workflow

1. **Data Loading & Splitting**
   - Loads colonoscopy image metadata.
   - Splits data into training and testing sets using `train_test_split` for reproducibility.

2. **DataLoader Construction**
   - Builds PyTorch `DataLoader` objects for both training and testing.
   - Combines augmented and original datasets to enhance generalization.

3. **Model Initialization**
   - Instantiates the segmentation model (`PolypFiLMNet`).
   - Allows configuration of sequence length and embedding dimensions.

4. **Training Setup**
   - **Loss Function:** Uses `BCEDiceLoss` (Binary Cross-Entropy + Dice loss).
   - **Optimizer:** AdamW with weight decay for regularization.
   - **Scheduler:** ReduceLROnPlateau for adaptive learning rate adjustment.

5. **Training & Evaluation**
   - Trains the model for a specified number of epochs.
   - Monitors metrics: **IoU** and **Dice coefficient**.
   - Saves model checkpoints and plots training history.

6. **Inference & Visualization**
   - Generates predictions on the test set.
   - Saves output masks and sample visualizations for qualitative assessment.

---

## Core Implementation

All steps are encapsulated in the `SegmentationPipeline` class, which exposes a single `run_full_pipeline()` method to execute the entire workflow seamlessly.

---

## Key Advantages

- **Modularity:** Each stage (loading, training, evaluation) is implemented as a separate function for easy maintenance and extension.
- **Reproducibility:** Fixed random seeds ensure consistent data splits and results.
- **Scalability:** Easily adaptable to other medical image segmentation datasets.
- **User-Friendly:** Run the entire pipeline with a single function call.

## Next Sections

- [Model Architecture](02_model_architecture.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
