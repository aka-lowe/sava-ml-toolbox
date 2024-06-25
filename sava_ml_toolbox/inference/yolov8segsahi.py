import logging
import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from sahi.models.yolov8onnx import Yolov8OnnxDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

from sava_ml_toolbox.inference.base import Model
from sava_ml_toolbox.utils import nms, sigmoid, xywh2xyxy
from sava_ml_toolbox.utils.runtime import ONNXRuntime

logger = logging.getLogger(__name__)


class Yolov8SegmModel(Yolov8OnnxDetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, num_masks=32, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
        """
        super().__init__(iou_threshold=iou_threshold, *args, **kwargs)

        self.num_masks = num_masks

    def _preprocess_image(
        self, image: np.ndarray, input_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        input_img = cv2.resize(image, input_shape)
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _post_process(
        self,
        outputs: np.ndarray,
        input_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
    ) -> List[np.array]:
        out = self._process_box_output(outputs[0], input_shape, image_shape)

        mask_maps = self._process_mask_output(
            out[1],
            outputs[1],
            [[elem[1], elem[0], elem[3], elem[2]] for elem in out[0]],
            image_shape,
        )
        self._original_masks = mask_maps
        return out[0]

    def _process_mask_output(
        self,
        mask_predictions: np.ndarray,
        mask_output: np.ndarray,
        boxes: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:

        if len(mask_predictions) == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(boxes, image_shape, (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), *image_shape))
        blur_size = (
            int(image_shape[1] / mask_width),
            int(image_shape[0] / mask_height),
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

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def _extract_boxes(
        self,
        box_predictions: np.ndarray,
        image_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
    ) -> np.ndarray:
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes,
            input_shape,
            image_shape,
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_shape[1])
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_shape[0])
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_shape[1])
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_shape[0])

        return boxes

    def _process_box_output(
        self,
        box_output: np.ndarray,
        input_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
    ) -> List[np.array]:
        # image_h, image_w = image_shape
        # input_w, input_h = input_shape

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = (
            self._extract_boxes(box_predictions, image_shape, input_shape)
            .round()
            .astype(np.int32)
        )

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        # Format the results
        prediction_result = []
        for bbox, score, label, mask in zip(
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        ):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append(
                [bbox[1], bbox[0], bbox[3], bbox[2], score, cls_id]
            )

        # prediction_result = [np.array(prediction_result), mask_predictions[indices]]

        return (np.array(prediction_result), np.array([mask_predictions[indices]]))

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        original_masks = self._original_masks

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, (
            image_predictions_in_xyxy_format,
            mask_prediction,
        ) in enumerate(zip([original_predictions], [original_masks])):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for i, prediction in enumerate(image_predictions_in_xyxy_format):
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction_list.append(
                    ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        bool_mask=np.array(mask_prediction, dtype=bool)[i],
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                )
                # del object_prediction
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        # Get input/output names shapes
        model_inputs = self.model.get_inputs()
        model_output = self.model.get_outputs()

        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_output[i].name for i in range(len(model_output))]

        input_shape = model_inputs[0].shape[2:]  # w, h
        image_shape = image.shape[:2]  # h, w

        # Prepare image
        image_tensor = self._preprocess_image(image, input_shape)

        # Inference
        outputs = self.model.run(
            input_data={input_names[0]: image_tensor}, output_names=output_names
        )

        # Post-process
        prediction_results = self._post_process(outputs, input_shape, image_shape)
        self._original_predictions = prediction_results

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


class YOLOv8SegSAHI(Model):
    def __init__(
        self,
        runtime: Union[ONNXRuntime, str],
        *args,
        iou_threshold: float = 0.5,
        slice_height: int = 256,
        slice_width: int = 256,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        num_masks: int = 32,
        conf_threshold: float = 0.5,
        **kwargs,
    ):
        assert (
            overlap_height_ratio > 0 and overlap_height_ratio < 1
        ), "overlap_height_ratio should be between 0 and 1"
        assert (
            overlap_width_ratio > 0 and overlap_width_ratio < 1
        ), "overlap_width_ratio should be between 0 and 1"
        assert (
            iou_threshold > 0 and iou_threshold < 1
        ), "iou_threshold should be between 0 and 1"
        assert (
            conf_threshold > 0 and conf_threshold < 1
        ), "conf_threshold should be between 0 and 1"

        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        self.model = Yolov8SegmModel(
            model=runtime,
            iou_threshold=iou_threshold,
            num_masks=num_masks,
            confidence_threshold=conf_threshold,
            *args,
            **kwargs,
        )

    def __call__(self, img_path, *args, **kwargs):
        result = get_sliced_prediction(
            img_path,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            *args,
            **kwargs,
        )
        boxes = [elem.bbox.to_xyxy() for elem in result.object_prediction_list]
        scores = [float(elem.score.value) for elem in result.object_prediction_list]
        class_ids = [int(elem.category.id) for elem in result.object_prediction_list]
        mask_maps = [
            np.array(elem.mask.bool_mask).astype(int)
            for elem in result.object_prediction_list
        ]

        return boxes, scores, class_ids, mask_maps

    def _inference(self):
        pass

    def _build_model(self):
        pass

    def _preprocessing(self):
        pass

    def _postprocessing(self):
        pass
