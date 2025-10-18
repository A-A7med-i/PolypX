# Training

The **Trainer** class manages the entire training, validation, and inference workflow for PolypX.
It provides a structured loop with logging, checkpointing, and evaluation.

---

## 1. Responsibilities of the Trainer

- **Training Loop**: Forward + Backward pass over the training data.
- **Validation Loop**: Evaluate model performance on unseen data.
- **Checkpointing**: Save the best model based on Dice score.
- **History Tracking**: Record metrics (Loss, IoU, Dice) per epoch.
- **Visualization**: Plot training curves and sample predictions.

---

## 2. Training Step

At each iteration, the trainer performs:

$$
\hat{y} = f_\theta(x, seq\_id)
$$

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{BCE} + \beta \cdot \mathcal{L}_{Dice}
$$

Model parameters are updated via:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

**Where:**

- $$\( x \)$$ → Input colonoscopy image
- $$\( seq\_id \)$$ → Sequence ID for FiLM modulation
- $$\( \hat{y} \)$$ → Predicted segmentation mask
- $$\( \theta \)$$ → Model parameters
- $$\( \eta \)$$ → Learning rate

---

## 3. Key Methods

### 🔹 `_run_epoch(training, dataloader)`

- Executes **one epoch** of training or validation.
- Computes **loss**, **IoU**, and **Dice score**.

---

### 🔹 `full_train()`

- Runs the complete loop for all epochs.
- Logs progress and saves **best model checkpoint**.
- Stores training history for visualization.

---

### 🔹 `plot_history(history)`

- Plots **Loss**, **IoU**, and **Dice** across epochs.
- Saves the figure to `history_path`.

---

### 🔹 `run_inference(test_loader)`

- Runs inference on unseen test data.
- Saves or displays visualizations of predictions vs ground truth.
- Produces side-by-side plots:
  - Original Image
  - Ground Truth Mask
  - Predicted Mask

---

## 4. Practical Notes

- The trainer automatically chooses **GPU if available**.
- Best checkpoint is determined by **highest Dice score** on validation.
- `ReduceLROnPlateau` scheduler lowers LR when validation loss plateaus.
- Predictions are thresholded at **0.5** after sigmoid activation.

---

## Next Sections

- [Pipeline Results](06_pipeline_results.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
