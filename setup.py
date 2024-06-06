from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="sava-ml-toolbox",
    version="0.1",
    description="ML Toolbox for SAVA project",
    author="PAS Group | DTU Elektro",
    author_email="fabmo@dtu.dk",
    packages=find_packages(),
    install_requires=required,
)
