import subprocess
import sys
import platform
import os

def is_aarch64():
    return platform.system() == "Linux" and "aarch64" in platform.machine()

def install_onnx_for_aarch64():
    print("Installing custom ONNX Runtime GPU version for aarch64...")
    url = "https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl"
    filename = "onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl"

    subprocess.check_call(["wget", url, "-O", filename])

    subprocess.check_call([sys.executable, "-m", "pip", "install", filename])

    os.remove(filename)

    print("ONNX Runtime GPU version for aarch64 installed successfully")


try:
    import setuptools
except ImportError:
    print("setuptools not found, installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as req_file:
    install_requires = req_file.read().splitlines()

if is_aarch64():

    # Remove onnxruntime from the requirements
    install_requires = [pkg for pkg in install_requires if "onnxruntime-gpu" not in pkg]

    install_onnx_for_aarch64() # Custom ONNX Runtime GPU version for aarch64

setuptools.setup(
    name="sava_ml_toolbox",
    version="0.2.0",
    author="PAS DTU Electro",
    author_email="fabmo@dtu.dk",
    description="2D/3D Object Detection Inference Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DTU-PAS/sava-ml-toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires=">=3.8",
)
