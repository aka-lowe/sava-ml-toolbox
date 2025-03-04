import os
from typing import Tuple

import numpy as np
import yaml
from PIL import Image

from sava_ml_toolbox.structures import DectObject, DetectionListResult
from sava_ml_toolbox.utils import draw_detections


class InferenceEngine:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        self.category_mapping = {
            str(list(d.keys())[0]): list(d.values())[0] for d in self.config["classes"]
        }
        # Instantiate the provider
        runtime = None
        if self.config["runtime"] == "onnxruntime":

            assert "model_path" in self.config, "model_path is required for onnxruntime"
            assert os.path.isfile(
                self.config["model_path"]
            ), "The model_path does not refer to a valid file."

            from sava_ml_toolbox.utils.runtime import ONNXRuntime

            runtime = ONNXRuntime(
                path=self.config["model_path"],
                providers=self.config["providers"],
            )

        elif self.config["runtime"] == "tensorrt":
            raise NotImplementedError("TensorRT runtime is not implemented yet.")

        else:
            raise ValueError(
                f"Invalid runtime {self.config['runtime']} specified in config file."
            )

        # Instantiate the wrapper
        if self.config["use_sahi_optim"]:
            from sava_ml_toolbox.inference import YOLOv8SegSAHI

            self.model = YOLOv8SegSAHI(
                runtime=runtime,
                iou_threshold=self.config["iou_thres"],
                slice_height=self.config["slice_height"],
                slice_width=self.config["slice_width"],
                overlap_width_ratio=self.config["overlap_width_ratio"],
                overlap_height_ratio=self.config["overlap_height_ratio"],
                num_masks=self.config["num_masks"],
                conf_threshold=self.config["conf_thres"],
                category_mapping=self.category_mapping,
            )

        else:
            from sava_ml_toolbox.inference import YOLOv8Seg

            self.model = YOLOv8Seg(
                runtime=runtime,
                conf_thres=self.config["conf_thres"],
                iou_thres=self.config["iou_thres"],
                num_masks=self.config["num_masks"],
            )

    def xyxy_to_xywh(self, boxes: np.array) -> np.array:
        """
        Convert bounding boxes from xyxy format to xywh format.

        Args:
            boxes (np.array): Bounding boxes in xyxy format with shape (N, 4).

        Returns:
            np.array: Bounding boxes in xywh format with shape (N, 4).
        """
        if len(boxes) == 0:
            return []
        xywh_boxes = np.zeros_like(boxes)
        xywh_boxes[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # center x
        xywh_boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # center y
        xywh_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
        xywh_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
        return xywh_boxes

    def predict(self, img: Image.Image) -> DetectionListResult:
        boxes, scores, class_ids, mask_maps = self.model(img)
        boxes = self.xyxy_to_xywh(boxes)
        out = DetectionListResult()
        for box, score, class_id, mask_map in zip(boxes, scores, class_ids, mask_maps):
            detection = DectObject(
                xywh=box, score=score, class_id=class_id, segm=mask_map
            )
            # detection.xywh = box
            # detection.score = score
            # detection.class_id = class_id
            # detection.segm = mask_map
            out.append(detection)

        return out

    def draw_results(self, img: Image.Image, results: DetectionListResult) -> np.array:
        out_img = draw_detections(
            np.array(img),
            results.getbboxes(),
            results.getscores(),
            results.getclassids(),
            self.category_mapping,
            0.4,
            results.getsegms(),
        )
        return out_img
