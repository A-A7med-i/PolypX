import torch.nn as nn
import torch


class IoU(nn.Module):
    """
    Compute the Intersection over Union (IoU) metric (a.k.a. Jaccard Index).

    IoU measures the ratio between the overlap and the union of the predicted
    segmentation and the ground truth mask.

    Args:
        smooth (float, optional):
            Small constant to avoid division by zero when union is zero.
            Default is 1.0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the IoU score.

        Args:
            logits:
                Raw, unnormalized model outputs of shape **[B, C, H, W]**.
            target:
                Ground truth binary masks of shape **[B, C, H, W]**.

        Returns:
            torch.Tensor:
                Scalar tensor containing the IoU score (range: 0–1).
        """
        probs = (torch.sigmoid(logits) > 0.5).float()

        target = target.float()

        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()

        union = probs.sum() + target.sum() - intersection

        iou_score = (intersection + self.smooth) / (union + self.smooth)

        return iou_score


class DiceMetric(nn.Module):
    """
    Compute the Dice Coefficient (a.k.a. Sørensen–Dice index / F1 score).

    Dice measures the overlap between predicted and ground truth masks,
    emphasizing correct positive predictions—especially useful for
    imbalanced datasets.

    Args:
        smooth (float, optional):
            Small constant to stabilize the calculation and prevent
            division by zero. Default is 1.0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Dice score.

        Args:
            logits:
                Raw, unnormalized model outputs of shape **[B, C, H, W]**.
            target:
                Ground truth binary masks of shape **[B, C, H, W]**.

        Returns:
            torch.Tensor:
                Scalar tensor containing the Dice score (range: 0–1).
        """
        probs = (torch.sigmoid(logits) > 0.5).float()

        target = (target > 0.5).float()

        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()

        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + target.sum() + self.smooth
        )

        return dice_score
