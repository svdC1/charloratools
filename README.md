## ![Alt text](https://res.cloudinary.com/ducarqe1h/image/upload/v1726127296/clt_banner_w5hpan.png)

### This is a simple python library which intends to **facilitate the process of image gathering,filtering and processing when creating a training dataset to use for training stable diffusion LoRas**. As the name suggests it is specifically useful when organizing a *character lora* image training dataset.



## Installation

<a href=https://pypi.org/project/charloratools/><img src=https://pypi.org/static/images/logo-small.8998e9d1.svg></img></a>

```bash
pip install charloratools
```
### Torch CPU-Only
```bash 
pip install charloratools[full]
```
## Installation Steps

### There are 2 dependencies that are not installed during installation through pip due to specific installation requirements for different OSes. Those are **PyTorch** and **facenet-pytorch**. This package won't work unless you **manually install those**.

## Dependency Install Script

### After installing with pip you can run a **script that automatically detects which distribution of those packages is the correct one for your current machine and installs them**.

### The script can also be used to check if you have the correct distribution of those dependencies installed or to fix their installation

### The script checks if you already have torch installed, but it will reinstall it if the version is incompatible with the rest of the package's dependencies.

## Installing and Running The Script

```bash
pip install charloratools
charloratools install_torch
```
## Installing Torch and facenet-pytorch manually

### For Torch, I recommend [following the steps described in their website](https://pytorch.org/get-started/locally/).

### For facenet-pytorch

```bash
pip install facenet-pytorch>=2.6.0
```


## Roadmap

### [Make functionality available as ComfyUI Custom Node](https://github.com/svdC1/comfy-ui-lora-dataset-tools)
