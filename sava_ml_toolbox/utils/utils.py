import numpy as np


def extract_patches(image: np.ndarray, kernel_size: tuple) -> np.ndarray:
    """Extract patches from an image.

    Args:
        image (np.ndarray): Input image.
        kernel_size (tuple): Size of each patch.

    Returns:
        np.ndarray: Array of extracted image patches.
    """
    h, w, c = image.shape
    th, tw = kernel_size

    # Reshape the image into tiles of specified size
    tiled_array = image.reshape(h // th, th, w // tw, tw, c)
    tiled_array = tiled_array.swapaxes(1, 2)  # Swap axes for correct tiling
    return tiled_array.reshape(-1, th, tw, c)


def pad_image_to_multiple(
    image: np.ndarray, kernel_size: int, pad_value: int = 0
) -> np.ndarray:
    """Pad an image to ensure its dimensions are multiples of a specified kernel size.

    Args:
        image (np.ndarray): Input image.
        kernel_size (int): Desired kernel size for padding.
        pad_value (int, optional): Value for padding. Defaults to 0.

    Returns:
        np.ndarray: Padded image.
    """
    H, W, _ = image.shape
    K = kernel_size
    pad_h = (K - (H % K)) % K
    pad_w = (K - (W % K)) % K

    # Pad the image to ensure dimensions are multiples of kernel_size
    padded_image = np.pad(
        image, ((0, pad_h), (0, pad_w), (pad_value, pad_value)), mode="constant"
    )

    return padded_image


def pad_batch(input_tensor: np.ndarray, target_batch_size: int) -> np.ndarray:
    """Pad a batch of input tensors to match the target batch size.

    Args:
        input_tensor (np.ndarray): Input tensor with shape (N, H, W, C).
        target_batch_size (int): Desired batch size.

    Returns:
        np.ndarray: Padded tensor with shape (B, H, W, C), where B is the target batch size.
    """
    N, H, W, C = input_tensor.shape
    B = target_batch_size

    if N >= B:
        return input_tensor

    empty_images = np.zeros((B - N, H, W, C), dtype=input_tensor.dtype)
    padded_tensor = np.concatenate([input_tensor, empty_images], axis=0)

    return padded_tensor


def ceiling_division(n: int, d: int) -> int:
    """Perform division with ceiling rounding.

    Args:
        n (int): Numerator.
        d (int): Denominator.

    Returns:
        int: Result of division with ceiling rounding.
    """
    assert isinstance(n, int), "n must be an integer"
    assert isinstance(d, int), "d must be an integer"
    assert d != 0, "d cannot be zero"
    return -(n // -d)
