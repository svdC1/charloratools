"""
> **Provides comprehensive tools for processing, filtering and managing
directories containing images, (i.e Image Datasets), present in the system.**

> Designed to **facilitate the process of manually separating, comparing and
performing other time-consuming tasks with image files** when cleaning an
image dataset.

> Originally created to speed up the process of cleaning
datasets used for Stable Diffusion LoRa training (hence the name),
**it leverages `facenet_pytorch`'s implementation of advanced face
detection and recognition ML models `(MTCNN and InceptionResnetv1)` and
`ImageHash`'s implementation of robust hashing algorithms for image files
to provide useful and accurate means of handling diverse image datasets.**

Key Features
------------
- **Facial Recognition**: Utilizes pre-trained models from `facenet_pytorch`
                          to provide robust, and easy-to-use, face detection
                          and recognition.

- **Image Comparison**: Allows choosing between the various hashing algorithms
                        implemented by ImageHash library (p-hash, d-hash,
                        avg_hash, crop-resistant-hash) to easily provide
                        more flexible means of comparing different images.

- **Social Media Scrapers - Beta**: Effortlessly scrape images from platforms
                                    like VSCO and Instagram, managing
                                    authentication and media retrieval.

- **Image Management**: Organize and filter images, automatically
                        delete corrupted image files, programatically
                        add and delete any image from a directory
                        with python operators, and other useful
                        functionalities.

- **Command-Line Interface**: Provides CLI utilities to easily install the
                              correct distribution of PyTorch by analyzing
                              the user's system (Wether CUDA or ROCm is
                              present, OS platform, etc...), avoiding
                              common torch installation errors.

- **Error Handling**: Custom exceptions to provide more information when an
                      error occurs during processing.


Examples
--------
**Install with pip**
```bash
pip install charloratools
```
**Run torch installation command**
```bash
charloratools install_torch
```
`The script automatically detects if torch is already installed and
skips installation if it is.`


Modules
-------
FilterAI
    Provides functionality for **face detection and recognition using
    the `facenet_pytorch`**.
SysFileManager
    Provides classes and functions for **managing and processing image
    files.**
Scrapers
    Provides **context manager classes for scraping and
    downloading media images from social media platforms.**
utils
    Provides utility functions for **image processing,
    facial recognition and scraping functionalities**
cli
    Provides command-line interface `(CLI)` utilities for **managing
    the installation of PyTorch with varying configurations**
errors
    Defines **custom exception classes used throughout the application
    to handle various types of errors**
"""


import importlib


def __getattr__(name):
    if name in ['cli', 'errors', 'FilterAI', 'Scrapers',
                'SysFileManager', 'utils', 'facenet_pytorch']:
        try:
            return importlib.import_module(f'.{name}', __name__)
        except ImportError as e:
            if 'torch' in str(e):
                raise ImportError(
                    f"""Submodule '{name}' requires 'torch' but it is
                        not installed.Please run 'charloratools install_torch'
                        to install it."""
                ) from None

            es = f"module '{__name__}' has no attribute '{name}'"

            raise AttributeError(es)
