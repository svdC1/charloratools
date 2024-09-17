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

### Torch is not installed during installation through pip due to specific installation requirements for different OSes. This package won't work unless you **manually install those**.

### You can also choose to install the OS independent version of torch ( uses only the CPU ) by running the "Torch CPU-Only" pip command listed above.

## Dependency Install Script

### After installing with pip you can run a **script that automatically detects which distribution of torch is the correct one for your current machine and installs it**.

### The script can also be used to check if you have the correct distribution of torch installed or to install the latest correct distribution for your OS.

### The script checks if you already have torch installed and quits without making any changes if you do.

## Installing and Running The Script

```bash
pip install charloratools
charloratools install_torch
```
## Installing Torch Manually

### If you don't want to run the script and prefer installing torch yourself I recommend [following the steps described in their website](https://pytorch.org/get-started/locally/). Once torch is installed the package should work correctly.

## ComfyUI Custom Node

### Some of the functionality of this package is already available as a custom node for ComfyUI [(comfy-ui-lora-dataset-tools)](https://github.com/svdC1/comfy-ui-lora-dataset-tools) , I'll be implementing the rest of the functionalities in new versions of the custom node.

## Roadmap

 - ### Create better documentation in wiki