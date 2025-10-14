from typing import List, Tuple, Union, Dict
from src.config.constant import FIG_SIZE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch


class Trainer:
    """
    Handles training, validation, and inference for a PyTorch segmentation model.

    This trainer provides:
        • A structured training/validation loop
        • Checkpoint saving for the best model
        • History plotting (loss, IoU, Dice)
        • Convenient inference utilities
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        checkpoint_path: Union[str, Path],
        history_path: Union[str, Path],
        loss_function: nn.Module,
        iou_metric: nn.Module,
        dice_metric: nn.Module,
    ):
        """
        Initialize the training manager.

        Args:
            model: PyTorch model to train.
            epochs: Total number of training epochs.
            optimizer: Optimizer for model parameters.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            scheduler: Learning rate scheduler (ReduceLROnPlateau recommended).
            checkpoint_path: Path to save the best model weights.
            history_path: Path to save the training history plot.
            loss_fn: Loss function (e.g., BCE + Dice).
            iou_metric: Intersection-over-Union metric for evaluation.
            dice_metric: Dice coefficient metric for evaluation.
        """
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.checkpoint_path = Path(checkpoint_path)
        self.history_path = Path(history_path)
        self.loss_function = loss_function
        self.iou_metric = iou_metric
        self.dice_metric = dice_metric
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.best_dice_acc = float("-inf")

    def _run_epoch(
        self, training: bool, dataloader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        Runs a single training or validation epoch.

        Args:
            training (bool): If True, runs a training epoch; otherwise, runs a validation epoch.
            dataloader (DataLoader): The DataLoader for the current epoch.

        Returns:
            Tuple[float, float, float]: The average loss, IoU, and Dice score for the epoch.
        """
        self.model.train() if training else self.model.eval()
        context = torch.enable_grad() if training else torch.inference_mode()
        running_loss, iou_acc, dice_acc, count = 0.0, 0.0, 0.0, 0

        desc = "Training" if training else "Testing"
        with context:
            for sequence_id, image, mask in tqdm(dataloader, desc=desc):
                sequence_id, image, mask = (
                    sequence_id.to(self.device),
                    image.to(self.device),
                    mask.to(self.device),
                )

                logits = self.model(image, sequence_id)
                loss = self.loss_function(logits, mask)
                iou_score = self.iou_metric(logits, mask)
                dice_score = self.dice_metric(logits, mask)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item()
                iou_acc += iou_score.item()
                dice_acc += dice_score.item()
                count += 1

        avg_loss = running_loss / count
        avg_iou = iou_acc / count
        avg_dice = dice_acc / count

        return avg_loss, avg_iou, avg_dice

    def plot_history(self, history: Dict[str, List[float]]):
        """
        Plots the training and validation history for loss, IoU, and Dice score.

        Args:
            history (Dict[str, List[float]]): A dictionary containing the metrics over epochs.
        """
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=FIG_SIZE)

        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(
            epochs,
            history["train_loss"],
            color="#2E86AB",
            marker="o",
            markersize=6,
            linestyle="-",
            linewidth=2,
            label="Train Loss",
        )
        plt.plot(
            epochs,
            history["test_loss"],
            color="#E27D60",
            marker="^",
            markersize=6,
            linestyle="--",
            linewidth=2,
            label="Test Loss",
        )
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Loss", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        # IOU
        plt.subplot(1, 3, 2)
        plt.plot(
            epochs,
            history["iou_train"],
            color="#28B463",
            marker="s",
            markersize=6,
            linestyle="-",
            linewidth=2,
            label="Train IOU",
        )
        plt.plot(
            epochs,
            history["iou_test"],
            color="#CA6F1E",
            marker="D",
            markersize=6,
            linestyle="--",
            linewidth=2,
            label="Test IOU",
        )
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("IOU", fontsize=12)
        plt.title("IOU", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        # DICE
        plt.subplot(1, 3, 3)
        plt.plot(
            epochs,
            history["dice_train"],
            color="#8E44AD",
            marker="*",
            markersize=7,
            linestyle="-",
            linewidth=2,
            label="Train DICE",
        )
        plt.plot(
            epochs,
            history["dice_test"],
            color="#F39C12",
            marker="v",
            markersize=6,
            linestyle="--",
            linewidth=2,
            label="Test DICE",
        )
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("DICE", fontsize=12)
        plt.title("DICE", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)

        plt.tight_layout()

        plt.savefig(self.history_path)
        print(f"Plots saved to {self.history_path}")

        plt.show()

    def full_train(self) -> Dict[str, List[float]]:
        """
        Executes the full training loop over a specified number of epochs.

        Returns:
            Dict[str, List[float]]: A dictionary containing the training history.
        """

        history = {
            "train_loss": [],
            "test_loss": [],
            "iou_train": [],
            "iou_test": [],
            "dice_train": [],
            "dice_test": [],
        }

        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")

            train_loss, iou_train, dice_train = self._run_epoch(
                training=True, dataloader=self.train_dataloader
            )
            test_loss, iou_test, dice_test = self._run_epoch(
                training=False, dataloader=self.test_dataloader
            )

            if dice_test > self.best_dice_acc:
                self.best_dice_acc = dice_test
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}")
                print(f"New best model with Dice Accuracy = {dice_test*100:.4f}%")

            history["train_loss"].append(train_loss)
            history["iou_train"].append(iou_train)
            history["test_loss"].append(test_loss)
            history["iou_test"].append(iou_test)
            history["dice_train"].append(dice_train)
            history["dice_test"].append(dice_test)

            print(
                f"Train Loss: {train_loss:.4f} | Train IOU: {iou_train*100:.2f}% | Train DICE: {dice_train*100:.2f}%"
            )

            print(
                f"Test Loss: {test_loss:.4f} | Test IOU: {iou_test*100:.2f}% | Test DICE: {dice_test*100:.2f}%"
            )

            self.scheduler.step(test_loss)

        return history

    def run_inference(
        self,
        test_loader: DataLoader,
        output_dir: str | Path | None = None,
        num_random_samples: int = 5,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Run inference on the test set and optionally save visualizations of random predictions.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
            output_dir (str | Path, optional): Directory to save random prediction images.
                If None, images are only displayed and not saved.
            num_random_samples (int, optional): Number of random images to visualize/save.

        Returns:
            Dict[str, List[np.ndarray]]: Inference results with keys:
                ``sequence_id``, ``image``, ``true_mask``, ``pred_mask``.
        """
        results: Dict[str, List[np.ndarray]] = {
            "sequence_id": [],
            "image": [],
            "true_mask": [],
            "pred_mask": [],
        }
        self.model.eval()

        with torch.inference_mode():
            for sequence_id, image, mask in tqdm(test_loader, desc="Running Inference"):
                sequence_id = sequence_id.to(self.device)
                image = image.to(self.device)
                mask = mask.to(self.device)

                # Get prediction.
                pred = self.model(image, sequence_id)
                mask_pred = torch.sigmoid(pred)
                binary_mask = (mask_pred > 0.5).float()

                # Store results on the CPU as numpy arrays.
                results["sequence_id"].append(sequence_id.cpu().numpy())
                results["image"].append(image.cpu().numpy())
                results["true_mask"].append(mask.cpu().numpy())
                results["pred_mask"].append(binary_mask.cpu().numpy())

        for i in range(num_random_samples):
            idx = np.random.randint(0, len(results["sequence_id"]))

            img = results["image"][idx][0]
            true_mask = results["true_mask"][idx][0][0]
            pred_mask = results["pred_mask"][idx][0][0]

            # Normalize & transpose for plotting
            img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min())

            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(14, 6))
            fig.suptitle(
                f"Sample {i + 1} – Green: Ground Truth | Red: Prediction",
                fontsize=16,
                fontweight="bold",
            )

            # Original Image
            axes[0].imshow(img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Ground Truth Mask
            axes[1].imshow(img)
            axes[1].imshow(true_mask, cmap="Greens", alpha=0.5)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Predicted Mask
            axes[2].imshow(img)
            axes[2].imshow(pred_mask, cmap="Reds", alpha=0.5)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.93])

            if output_dir:
                save_path = output_dir / f"prediction_sample_{i + 1}.png"
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
            else:
                plt.show()

        return results
