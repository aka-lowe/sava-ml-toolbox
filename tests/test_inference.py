import numpy as np
import yaml
from PIL import Image

from sava_ml_toolbox import DectObject, DetectionListResult, YOLOv8SegONNX
from sava_ml_toolbox.inference import InferenceEngine
from sava_ml_toolbox.utils import draw_detections

# Specify the path to your YAML file
CONFIG_FILE = "configs/yolo_segm_batch_1_size_640.yaml"


def test_yolov8seg_inference():
    # Load config
    with open(CONFIG_FILE, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load the sample image
    sample_image_path = "data/samples/sample5.jpg"
    img = Image.open(sample_image_path)

    model = YOLOv8SegONNX(
        model_path=config["model_path"],
        providers=config["providers"],
        # conf_threshold=config["conf_threshold"],
        patch_size=config["patch_size"],
        # batch_size=config["batch_size"],
    )

    boxes, scores, class_ids, mask_maps = model(np.array(img))

    out_img = draw_detections(
        np.array(img),
        boxes,
        scores,
        class_ids,
        0.4,
        mask_maps,
    )

    Image.fromarray((out_img).astype(np.uint8)).save("data/samples/pred_sample5.jpg")
    print("Model Inference Completed")


def test_inference_engine():
    config = "configs/yolo_segm_batch_1_size_640.yaml"
    # Load the sample image
    sample_image_path = "data/samples/sample5.jpg"
    img = Image.open(sample_image_path)

    # Create Inference Engine
    inference_engine = InferenceEngine(config)

    # Perform inference
    results = inference_engine.predict(np.array(img))

    out_img = draw_detections(
        np.array(img),
        results.getbboxes(),
        results.getscores(),
        results.getclassids(),
        0.4,
        results.getsegms(),
    )
    Image.fromarray((out_img).astype(np.uint8)).save("data/samples/pred_sample5.jpg")


if __name__ == "__main__":
    test_inference_engine()
