import numpy as np
import yaml
from PIL import Image

from sava_ml_toolbox import DectObject, DetectionListResult, YOLOV8SegModel

# Specify the path to your YAML file
CONFIG_FILE = "configs/yolo_od_batch_1_size_640.yaml"


def test_yolov8seg_inference():
    # Load config
    with open(CONFIG_FILE, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Load the sample image
    sample_image_path = "data/samples/sample5.jpg"
    img = Image.open(sample_image_path)

    runtime = YOLOV8SegModel(
        model_path=config["model_path"],
        providers=config["providers"],
        conf_threshold=config["conf_threshold"],
        patch_size=config["patch_size"],
        batch_size=config["batch_size"],
    )

    pred_points = runtime.inference(np.array(img))
    print("Model Inference Completed")


if __name__ == "__main__":
    test_yolov8seg_inference()
