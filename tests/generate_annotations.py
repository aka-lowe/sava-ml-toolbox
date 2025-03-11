import json
import os

import fiftyone.zoo as foz
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm

from sava_ml_toolbox.inference import InferenceEngine
from sava_ml_toolbox.utils._OLD_evaluation import mAPF1Metrics

# Define the paths for the dataset
data_dir = "data"

# Specify the path to your YAML file
CONFIG_FILE_SMALL = "configs/yolov8n_onnx.yaml"
# CONFIF_FILE_BIG = "configs/yolov8_onnx.yaml"

# # Download the COCO 2017 dataset using fiftyone
# dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_dir=data_dir)

transform = transforms.Compose([transforms.ToTensor()])


def create_json_output(model, loader):
    # Initialize the metric
    with open("./data/ultralitics-python.json", "r") as f:
        cat_conv = json.load(f)
    cat_conv = {int(key): int(cat_conv[key]) for key in cat_conv}
    pred_annotations = []

    for img, targets in tqdm(loader):
        if len(targets) == 0:
            continue
        img_np = (img.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        # Perform inference
        outputs = model.predict(img_np)
        coco_out = outputs.getcocoformat()
        img_id = targets[0]["image_id"]
        # Process predicted annotations
        for pred in coco_out:
            # Convert numpy array to list and handle other fields
            pred_bbox = pred["bbox"].tolist()
            pred_category_id = int(pred["category_id"])
            pred_score = float(pred["score"])

            annotation = {
                "image_id": int(img_id),
                "bbox": pred_bbox,
                "category_id": cat_conv[pred_category_id],
                "score": pred_score,
                "area": pred_bbox[2] * pred_bbox[3],
            }
            pred_annotations.append(annotation)

    with open("./data/output.json", "w") as fout:
        json.dump(pred_annotations, fout)


if __name__ == "__main__":
    model = InferenceEngine(CONFIG_FILE_SMALL)
    # big_model = InferenceEngine(CONFIG_FILE)
    # Load the validation dataset
    val_dataset = CocoDetection(
        root=os.path.join(data_dir, "images/val2017"),
        annFile=os.path.join(data_dir, "instances_val2017.json"),
        transform=transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    create_json_output(model, val_loader)
