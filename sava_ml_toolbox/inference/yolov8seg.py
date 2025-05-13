# Readapted the code from https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation/blob/main/image_instance_segmentation.py
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from sava_ml_toolbox.utils import draw_detections, nms, sigmoid, xywh2xyxy
from sava_ml_toolbox.utils.runtime import ONNXRuntime, TensorRTRuntime

from .base import Model


class YOLOv8Seg(Model):
    """
    This class represents a YOLOv8 model for object detection and instance segmentation, loaded from an ONNX file. It uses the ONNXRuntime for inference.

    The class provides methods for building the model, performing inference on an image, and processing the output to return bounding boxes, class scores, class IDs, and mask maps.

    Attributes:
        - model_path: Path to the ONNX model file.
        - providers: List of providers for the ONNXRuntime, default is ["CPUExecutionProvider"].
        - patch_size: Size of the patches to be used for the model, default is 640.
        - conf_threshold: Confidence threshold for filtering detections, default is 0.7.
        - iou_threshold: Intersection over Union (IoU) threshold for non-maximum suppression, default is 0.5.
        - num_masks: Number of masks used in the model, default is 32.
        - mean: Mean values for normalization, default is [0.485, 0.456, 0.406].
        - std: Standard deviation values for normalization, default is [0.229, 0.224, 0.225].
        - session: ONNXRuntime session for running the model.

    Outputs of the __call__() function:
        - boxes: Bounding boxes for detected objects.
        - scores: Confidence scores for detected objects.
        - class_ids: Class IDs for detected objects.
        - mask_maps: Mask maps for instance segmentation.
    """

    def __init__(
        self,
        runtime: Optional[BaseRuntime],
        patch_size: int = 640,
        conf_thres=0.7,
        iou_thres=0.5,
        num_masks=32,
    ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.patch_size = patch_size
        self.session = runtime
        self._get_input_details()
        self._get_output_details()

    def __call__(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.segment_objects(np.array(image))

    def segment_objects(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self._preprocessing(image)
        outputs = self._inference(input_tensor)
        boxes, scores, class_ids, mask_maps = self._postprocessing(outputs)
        return boxes, scores, class_ids, mask_maps

    def _postprocessing(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boxes, scores, class_ids, mask_pred = self._process_box_output(outputs[0])
        mask_maps = self._process_mask_output(mask_pred, outputs[1], boxes)
        return boxes, scores, class_ids, mask_maps

    def _preprocessing(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]
        # input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(image, (self.input_height, self.input_width))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        
        input_feed = {self.input_names[0]: input_tensor}

        outputs = self.session.run(input_data=input_feed, output_names=self.output_names if hassattr(self.session, 'ort_session') else None)
        return outputs

    def _process_box_output(
        self, box_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self._extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        )

    def _process_mask_output(
        self, mask_predictions: np.ndarray, mask_output: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (
            int(self.img_width / mask_width),
            int(self.img_height / mask_height),
        )
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )
            # if blur_size[0] == 0 or blur_size[1] == 0:
            #     blur_size = (1, 1)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def _extract_boxes(self, box_predictions: np.ndarray) -> np.ndarray:
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def _get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(
        boxes: np.ndarray, input_shape: Tuple[int, int], image_shape: Tuple[int, int]
    ) -> np.ndarray:
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )

        return boxes
