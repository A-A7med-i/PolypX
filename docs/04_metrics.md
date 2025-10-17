# Evaluation Metrics

<div align="center">
  <img src="https://lh3.googleusercontent.com/proxy/pRWCsWtnp7rCksUp98PGRpuu4829n4Pz92o3L4XHw3Ay6bWo6GuxG0JZGeBhTJL4-dKSzU9BnWJYilSOYQcwztSCEMKSAfyaqwi6GcivUITCsTgV"
       alt="IoU and Dice visualization"
       width="800"
       height="300"
       />
</div>

---

Accurate evaluation of segmentation models requires metrics that measure the quality of predicted masks compared to the ground truth.
In **PolypX**, we use two widely adopted metrics: **IoU (Intersection over Union)** and **Dice Coefficient**.

---

## 1. Intersection over Union (IoU)

The **IoU**, also known as the *Jaccard Index*, measures the ratio of overlap to union between predicted and ground truth masks:

$$
IoU = \frac{|P \cap G|}{|P \cup G|}
$$

**Where:**

- \( P \) → Predicted mask
- \( G \) → Ground truth mask

---

### 🔹 IoU Formula in terms of pixels

$$
IoU = \frac{TP}{TP + FP + FN}
$$

**Where:**

- \( TP \) → True Positives
- \( FP \) → False Positives
- \( FN \) → False Negatives

---

## 2. Dice Coefficient

The **Dice Coefficient** (a.k.a. *Sørensen–Dice index* or *F1 score*) measures the harmonic mean between precision and recall, emphasizing overlap:

$$
Dice = \frac{2 \cdot |P \cap G|}{|P| + |G|}
$$

---

### 🔹 Dice Formula in terms of pixels

$$
Dice = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

**Where:**

- \( TP \) → True Positives
- \( FP \) → False Positives
- \( FN \) → False Negatives

---

## 3. Practical Notes

- Both IoU and Dice range from **0 (no overlap)** to **1 (perfect overlap)**.
- **IoU** penalizes false positives and false negatives more strongly.
- **Dice** is more robust when dealing with **imbalanced datasets** (e.g., small polyp regions).
- Using both metrics together provides a **comprehensive evaluation** of model performance.

---

## Next Sections

- [Training](05_training.md)

*Continue to the next sections for deeper technical details on each pipeline component.*
