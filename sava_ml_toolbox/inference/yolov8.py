from typing import List, Optional

import numpy as np
from PIL import Image

from sava_ml_toolbox.structures import DetectionListResult
from sava_ml_toolbox.utils import (
    ceiling_division,
    extract_patches,
    pad_batch,
    pad_image_to_multiple,
)
from sava_ml_toolbox.utils.runtime import ONNXRuntime

from .base import Model


class YOLOV8SegModel(Model):
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
        conf_threshold: float = 0.5,
        patch_size: int = 640,
        batch_size: int = 1,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        super(Model, self).__init__()

        assert isinstance(model_path, str), "model_path must be a string"
        assert isinstance(providers, list), "providers must be a list"
        assert all(
            isinstance(provider, str) for provider in providers
        ), "providers must contain strings"
        assert isinstance(conf_threshold, float), "head_threshold must be a float"
        assert isinstance(patch_size, int), "patch_size must be an int"
        assert isinstance(batch_size, int), "batch_size must be an int"
        assert (
            isinstance(mean, list) and len(mean) == 3
        ), "mean must be a list of 3 floats"
        assert all(
            isinstance(value, float) for value in mean
        ), "mean values must be floats"
        assert isinstance(std, list) and len(std) == 3, "std must be a list of 3 floats"
        assert all(
            isinstance(value, float) for value in std
        ), "std values must be floats"

        self.batch_size = batch_size
        self.mean, self.std = mean, std

        self.conf_threshold = conf_threshold
        self.patch_size = patch_size

    def _normalize_images(self, images: np.ndarray) -> np.ndarray:
        """Normalize a batch of images for inference.

        Args:
            images (np.ndarray): Input batch of images with shape (N, H, W, C).

        Returns:
            np.ndarray: Normalized batch of images with shape (N, C, H, W).
        """
        images /= 255.0
        images = (images - self.mean) / self.std
        images = np.transpose(images, (0, 3, 1, 2))
        return images

    def _preprocessing(self, image: Image) -> np.array:
        """Preprocess the input image for inference.

        Args:
            image (Image): The input image.

        Returns:
            np.array: Preprocessed image as a NumPy array.
        """

        image = np.array(image).astype(np.float32)
        image = pad_image_to_multiple(image, self.patch_size, 0)
        patches = extract_patches(image, (self.patch_size, self.patch_size))
        patches = self._normalize_images(patches)
        return patches

    def _build_model(
        self,
        model_path: str,
        providers: Optional[List[str]] = ["CPUExecutionProvider"],
    ) -> None:
        """Build the ONNX Runtime model.

        Args:
            model_path (str): Path to the ONNX model file.
            providers (Optional[List[str]]): List of execution providers for ONNX Runtime.
        """
        self.runtime = ONNXRuntime(model_path, providers)

    def _postprocessing(self):
        # add your implementation here
        pass

    def inference(self, image: Image) -> DetectionListResult:
        """Perform inference on the input image.

        Args:
            image (Image): The input image.

        Returns:
            DetectionListResult: Object containing all results (2D/3D bbox, segmentation, point cloud).
        """
        # Preprocess the input image
        images = self._preprocessing(image)

        # Initialize variables to store predicted logits and points
        pred_logits_batch = pred_points_batch = None

        # Loop over image batches for inference
        for i in range(ceiling_division(images.shape[0], self.batch_size)):
            batch = images[i * self.batch_size : (i + 1) * self.batch_size, :, :]

            # Pad the batch if its size is less than the specified batch size
            if batch.shape[0] < self.batch_size:
                batch = pad_batch(batch, self.batch_size)

            # Run inference using the ONNX Runtime model
            pred_logits, pred_points = self.runtime.run(batch)

            # Concatenate the predictions to the accumulated batch
            pred_logits_batch = (
                np.concatenate([pred_logits_batch, pred_logits], axis=0)
                if pred_logits_batch is not None
                else pred_logits
            )

            pred_points_batch = (
                np.concatenate([pred_points_batch, pred_points], axis=0)
                if pred_points_batch is not None
                else pred_points
            )

        # Postprocess the results and return head localization
        return self._postprocessing(pred_points_batch, pred_logits_batch, image.shape)
