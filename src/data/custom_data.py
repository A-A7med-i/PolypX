from src.entities.models import LoadedImageData
from src.utils.helper import build_transforms
from torch.utils.data import Dataset
from typing import List, Tuple
import torch


class ClinicDataset(Dataset):
    """
    A PyTorch Dataset for clinical image segmentation tasks.

    This dataset handles raw image/mask pairs and their sequence IDs,
    applies augmentations, and outputs tensors ready for training
    or evaluation.

    Args:
        data (List[LoaderData]):
            A list of LoaderData objects containing:
                - sequence_id (int): The sequence identifier.
                - image (np.ndarray): The raw RGB image (H, W, 3).
                - mask (np.ndarray): The corresponding binary mask (H, W).
        is_train (bool):
            Whether to apply training augmentations (`True`) or
            validation/test preprocessing (`False`).
    """

    def __init__(self, data: List[LoadedImageData], is_train: bool):
        self.data = data
        self.transforms = build_transforms(is_train)

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single transformed sample.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - **sequence_id_tensor** (`torch.Tensor`, shape: [1]):
                    Sequence ID as a long tensor.
                - **image_tensor** (`torch.Tensor`, shape: [3, H, W]):
                    Augmented image as a float tensor.
                - **mask_tensor** (`torch.Tensor`, shape: [1, H, W]):
                    Augmented binary mask as a float tensor
                    with an added channel dimension.
        """
        item = self.data[index]

        image, mask = item.image, item.mask

        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].unsqueeze(0).float().contiguous()

        sequence_id_tensor = torch.tensor(item.sequence_id, dtype=torch.long)

        return sequence_id_tensor, image_tensor, mask_tensor
