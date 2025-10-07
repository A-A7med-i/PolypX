from albumentations.pytorch import ToTensorV2
from src.config.constant import NEW_SIZE
import albumentations as A


def build_transforms(is_train: bool):
    """
    Build an Albumentations transformation pipeline for training or validation.

    During training, a variety of augmentations are applied to increase
    dataset diversity and improve model generalization. During validation
    or testing, only resizing and normalization are applied.

    Args:
        is_train (bool):
            - True: Apply strong data augmentation (flips, rotations, noise, etc.).
            - False: Apply only deterministic preprocessing (resize + normalize).

    Returns:
        A.Compose:
            An Albumentations `Compose` object that can be called like
            `transform(image=image, mask=mask)` to obtain transformed tensors.

    Example:
        >>> transform = build_transforms(is_train=True)
        >>> out = transform(image=img, mask=mask)
        >>> img_t, mask_t = out["image"], out["mask"]
    """
    if is_train:
        augmentations = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            A.MotionBlur(blur_limit=3, p=0.25),
            A.Resize(*NEW_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ]
    else:
        augmentations = [A.Resize(*NEW_SIZE), A.Normalize(), ToTensorV2()]
    return A.Compose(augmentations)
