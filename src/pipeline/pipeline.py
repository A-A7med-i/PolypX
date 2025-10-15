from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from src.metrics.metrics import IoU, DiceMetric
from src.data.custom_data import ClinicDataset
from src.model.model import PolypFiLMNet
from src.training.trainer import Trainer
from src.loss.loss import BCEDiceLoss
from typing import Tuple, Dict, Any
from src.data.loader import Loader
import torch


class SegmentationPipeline:
    """
    A complete training pipeline for medical image segmentation tasks.

    This class encapsulates:
        1. Data loading and splitting
        2. DataLoader creation
        3. Model initialization
        4. Loss, optimizer, and scheduler setup
        5. Model training and evaluation
    """

    def __init__(
        self,
        meta_data: str,
        test_size: float,
        random_state: int,
        batch_size: int,
        num_sequences: int,
        emb_dim: int,
        lr: float,
        weight_decay: float,
        mode: str,
        factor: float,
        patience: int,
        epochs: int,
        checkpoint_path: str,
        history_path: str,
        output_dir: str | None = None,
    ) -> None:
        """
        Initialize the pipeline with configuration parameters.

        Args:
            meta_data: Path or identifier for dataset metadata.
            test_size: Proportion of data used for testing.
            random_state: Random seed for reproducibility.
            batch_size: Number of samples per DataLoader batch.
            num_sequences: Number of image sequences for the model.
            emb_dim: Embedding dimension for the model.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for AdamW optimizer.
            mode: Mode for ReduceLROnPlateau ('min' or 'max').
            factor: Factor by which the learning rate will be reduced.
            patience: Number of epochs to wait before reducing LR.
            epochs: Total number of training epochs.
            checkpoint_path: Path to save model checkpoints.
            history_path: Path to save training history plots.
            output_dir: Directory to save predicted mask plots and inference visualizations.
        """
        self.meta_data = meta_data
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.emb_dim = emb_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.output_dir = output_dir

        # Will be populated later
        self.train_data = None
        self.test_data = None
        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.model: torch.nn.Module | None = None
        self.loss_fn = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = None

    def load_data(self) -> Tuple[Any, Any]:
        """
        Load all images and split into train/test sets.

        Returns:
            Tuple of (train_data, test_data).
        """
        loader = Loader(self.meta_data)
        data = loader.load_all_data()
        self.train_data, self.test_data = train_test_split(
            data, test_size=self.test_size, shuffle=True, random_state=self.random_state
        )
        return self.train_data, self.test_data

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test DataLoaders.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        assert (
            self.train_data is not None and self.test_data is not None
        ), "Call load_and_split_data() first."

        train_augmented = ClinicDataset(self.train_data, is_train=True)
        train_plain = ClinicDataset(self.train_data, is_train=False)
        train_dataset = ConcatDataset([train_augmented, train_plain])

        test_dataset = ClinicDataset(self.test_data, is_train=False)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return self.train_loader, self.test_loader

    def build_model(self) -> torch.nn.Module:
        """
        Initialize the segmentation model.

        Returns:
            A PolypFiLMNet instance.
        """
        self.model = PolypFiLMNet(
            num_sequences=self.num_sequences, emb_dim=self.emb_dim
        )
        return self.model

    def setup_training_components(
        self,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
        """
        Setup loss function, optimizer, and learning rate scheduler.

        Returns:
            Tuple of (loss_fn, optimizer, scheduler).
        """
        assert self.model is not None, "Call build_model() first."

        self.loss_fn = BCEDiceLoss()
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=self.mode, factor=self.factor, patience=self.patience
        )
        return self.loss_fn, self.optimizer, self.scheduler

    def train_and_evaluate(self) -> Tuple[Dict[str, list], Any]:
        """
        Train the model and evaluate on the test set.

        Returns:
            history: Training history dictionary (loss, metrics, etc.)
            test_results: Inference results on the test set.
        """
        assert all(
            [
                self.model,
                self.loss_fn,
                self.optimizer,
                self.scheduler,
                self.train_loader,
                self.test_loader,
            ]
        ), "Make sure to run all setup methods first."

        trainer = Trainer(
            model=self.model,
            epochs=self.epochs,
            optimizer=self.optimizer,
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            scheduler=self.scheduler,
            checkpoint_path=self.checkpoint_path,
            history_path=self.history_path,
            loss_function=self.loss_fn,
            iou_metric=IoU(),
            dice_metric=DiceMetric(),
        )

        history = trainer.full_train()
        trainer.plot_history(history)
        test_results = trainer.run_inference(
            test_loader=self.test_loader,
            output_dir=self.output_dir,
            num_random_samples=5,
        )
        return history, test_results

    def run_full_pipeline(self) -> Tuple[Dict[str, list], Any]:
        """
        Execute the entire training pipeline in a single call.

        Returns:
            history: Training history dictionary.
            test_results: Inference results on the test set.
        """
        self.load_data()
        self.create_dataloaders()
        self.build_model()
        self.setup_training_components()
        return self.train_and_evaluate()
