from dataclasses import dataclass
import numpy as np


@dataclass
class LoadedImageData:
    """
    Container for a single image-mask pair with its sequence identifier.

    Attributes:
        sequence_id (int): Unique identifier for the data sequence.
        image (np.ndarray): Loaded RGB image array (H, W, 3), dtype=uint8.
        mask (np.ndarray): Binary mask array (H, W), dtype=uint8 (values 0 or 1).
    """

    sequence_id: int
    image: np.ndarray
    mask: np.ndarray
