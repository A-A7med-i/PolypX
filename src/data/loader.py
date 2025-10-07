from src.entities.models import LoadedImageData
from src.config.constant import BASE_DIR
from typing import Iterator, List
from tqdm.auto import tqdm
from pathlib import Path
import polars as pl
import numpy as np
import cv2


class Loader:
    """
    Loader for images and masks defined by a metadata CSV file.

    The metadata CSV must contain columns:
        - "sequence_id": Unique identifier for each sample
        - "image_path" : Relative path to the image file
        - "mask_path"  : Relative path to the mask file

    This class provides two main loading strategies:
        1. `load_all_data`: Load all samples into memory (simple but memory-heavy).
        2. `data_generator`: Stream samples one by one (memory-efficient).
    """

    def __init__(self, metadata_path: str | Path):
        """
        Initialize the loader and read the CSV metadata.

        Args:
            metadata_path (str | Path):
                Path to the CSV file containing image and mask file paths.
        """
        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pl.DataFrame:
        """
        Load and return the metadata as a Polars DataFrame.

        Returns:
            pl.DataFrame: DataFrame with columns
                        ["sequence_id", "image_path", "mask_path"].
        """
        return pl.read_csv(self.metadata_path)

    def load_all_data(self) -> List[LoadedImageData]:
        """
        Load all images and masks into memory.

        Warning:
            This method may consume a lot of RAM for large datasets.
            Use `data_generator` instead for large-scale data.

        Returns:
            List[LoadedImageData]: List of fully loaded samples.
        """
        return list(self.data_generator())

    def data_generator(self) -> Iterator[LoadedImageData]:
        """
        Stream images and masks one at a time.

        Yields:
            LoadedImageData:
                Object containing sequence ID, RGB image, and binary mask.
        """
        total_rows = self.metadata.height

        for row in tqdm(
            self.metadata.iter_rows(named=True),
            total=total_rows,
            desc="Loading Images",
        ):
            img_path = BASE_DIR / row["image_path"]
            mask_path = BASE_DIR / row["mask_path"]

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            yield LoadedImageData(
                sequence_id=int(row["sequence_id"]), image=image, mask=mask
            )
