import numpy as np
from PIL import Image
from sahi.predict import get_sliced_prediction

from sava_ml_toolbox.inference import InferenceEngine
from sava_ml_toolbox.utils import draw_detections

# Specify the path to your YAML file
CONFIG_FILE = "configs/yolov8_onnx.yaml"


def test_inference_engine():
    for i in range(1, 7):
        # Load the sample image
        sample_image_path = (
            f"/home/fabmo/works/sava-ml-toolbox/data/samples/sample{i}.jpg"
        )

        img = Image.open(sample_image_path)

        # Create Inference Engine
        inference_engine = InferenceEngine(CONFIG_FILE)

        # Perform inference
        results = inference_engine.predict(img)

        # Draw the results
        out_img = inference_engine.draw_results(img, results)

        Image.fromarray((out_img).astype(np.uint8)).save(
            sample_image_path.replace(".jpg", "_pred.jpg")
        )


if __name__ == "__main__":
    test_inference_engine()
