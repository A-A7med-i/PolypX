# Loss Function

---

## 1. Binary Cross-Entropy (BCE) Loss

The **BCE Loss** measures the pixel-wise classification error:

$$
\mathcal{L}_{BCE} = - \frac{1}{N} \sum_{i=1}^{N}
\Big[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \Big]
$$

**Where:**

- $y_i$ → Ground truth label for pixel $i$
- $\hat{y}_i$ → Predicted probability for pixel $i$
- $N$ → Total number of pixels

---

## 2. Dice Loss

The **Dice Loss** evaluates the overlap between prediction and ground truth:

$$
\mathcal{L}_{Dice} = 1 - \frac{2 \cdot \sum_{i=1}^{N} y_i \hat{y}_i + \epsilon}
{\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} \hat{y}_i + \epsilon}
$$

**Where:**

- $y_i$ → Ground truth pixel
- $\hat{y}_i$ → Predicted pixel probability
- $\epsilon$ → Small smoothing factor to avoid division by zero

---

## 3. Combined Loss (BCEDiceLoss)

The final hybrid loss is a weighted sum:

$$
\mathcal{L}_{BCEDice} = \alpha \cdot \mathcal{L}_{BCE} + \beta \cdot \mathcal{L}_{Dice}
$$

**Where:**

- $\alpha$ → Weight for **BCE loss** (default $\alpha = 0.5$)
- $\beta$ → Weight for **Dice loss** (default $\beta = 0.5$)

---

## 4. Practical Notes

- Handles **class imbalance** by emphasizing small polyp regions
- Stable with **AdamW optimizer** + LR scheduling
- Dice ensures **region-level accuracy**, not just pixel-level
- Common setup: $\alpha = \beta = 0.5$

---

## 5. Why BCEDiceLoss?

- Handles **class imbalance** better than BCE alone.
- Encourages both **pixel-level accuracy** and **region-level overlap**.
- Well-suited for **medical segmentation tasks** like polyp detection.

## Next Sections

- [Metrics](04_metrics.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
