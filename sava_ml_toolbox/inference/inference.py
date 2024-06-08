from typing import Tuple

import numpy as np
import yaml

from sava_ml_toolbox.structures import DectObject, DetectionListResult


class InferenceEngine:
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        if self.config["runtime"] == "onnxruntime":
            from sava_ml_toolbox.inference import YOLOv8SegONNX

            self.model = YOLOv8SegONNX(
                model_path=self.config["model_path"],
                providers=self.config["providers"],
                conf_thres=self.config["conf_thres"],
                iou_thres=self.config["iou_thres"],
                num_masks=self.config["num_masks"],
            )

    def predict(self, img: np.ndarray) -> DetectionListResult:
        boxes, scores, class_ids, mask_maps = self.model(img)
        out = DetectionListResult()
        for box, score, class_id, mask_map in zip(boxes, scores, class_ids, mask_maps):
            detection = DectObject(
                xywh=box, score=score, class_id=class_id, segm=mask_map
            )
            detection.xywh = box
            detection.score = score
            detection.class_id = class_id
            detection.segm = mask_map
            out.append(detection)

        return out
