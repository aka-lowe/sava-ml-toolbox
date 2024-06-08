# SAVA ML TOOLBOX

SAVA ML TOOLBOX is a Python library for Machine Learning tasks. It provides a collection of tools and utilities to simplify the process of deploying machine learning models for SAVA project.

## Installation

### Clone the repository

```bash
git clone https://github.com/username/sava-ml-toolbox.git
```
And navigate into the folder
```bash
cd sava-ml-toolbox
```

### Create environment (Optional)

Create a virtual environment in python with `venv`:
```bash
python3 -m venv .env
``` 
And activate it on Windows:
```bash 
.env\Scripts\activate 
```
And activate it on Linux/Mac:
```bash 
source .env/bin/activate 
```
### Install the library
Navigate to the root directory of the project (where the `setup.py` file is located), and run:
```bash 
pip install -e .
```
## Usage
Pick one of the configurations from the `config` folder or create one in YAML format, with the following template:

```yaml
model_path: <path_to_your_model>
runtime: <runtime_provider>
providers:
  - <provider_1>
  - <provider_2>
conf_threshold: <confidence_threshold>
batch_size: <batch_size>
conf_thres: <confidence_threshold>
iou_thres: <iou_threshold>
num_masks: <number_of_masks>
```

Then you can use the Inference Engine as follow:
```python
from sava_ml_toolbox.inference import InferenceEngine
from sava_ml_toolbox.utils import draw_detections
# ...

config = "path/to/config.yaml"

# Create Inference Engine
inference_engine = InferenceEngine(config)

# Load the sample image as numpy array
# ...


# Perform inference
results = inference_engine.predict(img)

# Print the results
out_img = draw_detections(
    np.array(img),
    results.getbboxes(),
    results.getscores(),
    results.getclassids(),
    0.4,
    results.getsegms(),
)
# ...
```

The output of the `predict` function is a class which contains the following attributes:

- `xywh` - 2D bounding box
- `xyzwhd` - 3D bounding box
- `segm` - Segmentation mask
- `class_id`  - Class ID of the detected object
- `score` - Confidence score of the detection
- `pc`  - Point cloud data

## Export YOLO family to ONNX

After installing ultralitics through pip, the following command allows to export yolo models:
```bash
yolo export model=yolov8x-seg.pt format=onnx
```


## License

