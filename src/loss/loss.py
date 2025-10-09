import torch.nn as nn
import torch


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.

    The Dice coefficient measures the overlap between predicted and
    ground-truth masks. This loss is particularly effective for
    imbalanced datasets by focusing on the overlap rather than
    pixel-wise classification alone.

    Args:
        smooth (float, optional):
            Smoothing factor to avoid division by zero
            and stabilize gradients. Default is ``1.0``.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Args:
            logits (torch.Tensor):
                Raw, unnormalized model outputs of shape ``[B, C, H, W]``.
            target (torch.Tensor):
                Ground-truth binary masks of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor:
                Scalar tensor containing the Dice loss.
        """
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()

        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum() + target.sum() + self.smooth
        )

        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Dice loss.

    This loss function balances:
        - **BCE**: Penalizes pixel-wise classification errors.
        - **Dice**: Encourages overlap between prediction and ground truth.

    Combining them captures both local (per-pixel) and global
    (region-level) performance.

    Args:
        smooth (float, optional):
            Smoothing factor passed to the underlying Dice loss.
            Default is ``1.0``.
    """

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined BCE + Dice loss.

        Args:
            logits (torch.Tensor):
                Raw, unnormalized model outputs of shape ``[B, C, H, W]``.
            target (torch.Tensor):
                Ground-truth binary masks of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor:
                Scalar tensor representing the total loss (``BCE + Dice``).
        """
        return self.dice_loss(logits, targets) + self.bce_loss(logits, targets)
