import subprocess
import sys
import ensurepip
import logging
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

ensurepip.bootstrap()
# Function to detect CUDA version
def get_cuda_version():
  try:
    # Check if nvcc is installed (CUDA compiler)
    output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
    for line in output.split("\n"):
      if "release" in line:
        # Extract version from the line like: "release 10.2, V10.2.89"
        version = line.split("release")[-1].strip().split(",")[0]
        return version
  except Exception as e:
    logging.warning(f"Could not determine CUDA version: {e}")
  return None

# Function to install PyTorch and related packages based on CUDA version
def install_pytorch(cuda_version):
  if cuda_version:
    logging.info(f"Detected CUDA version: {cuda_version}")
    if cuda_version.startswith("12"):
      # Install for CUDA 12.x
      subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
				"xformers",
        "--index-url", "https://download.pytorch.org/whl/cu121"
      ])
      subprocess.check_call([sys.executable,"-m",'pip','install','facenet_pytorch','--no-deps'])
    elif cuda_version.startswith("11"):
      # Install for CUDA 11.x
      subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
				"xformers",
        "--index-url", "https://download.pytorch.org/whl/cu118"
      ])
      subprocess.check_call([sys.executable,"-m",'pip','install','facenet_pytorch','--no-deps'])
    else:
      # Fall back to CPU if no matching CUDA version
      logging.warning("No suitable CUDA version found, falling back to CPU version of PyTorch.")
      subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.2.2",
        "torchvision==0.17.2",
        "torchaudio==2.2.2",
      ])
      subprocess.check_call([sys.executable,"-m",'pip','install','facenet_pytorch','--no-deps'])
  else:
    # Fall back to CPU if CUDA is not available
    logging.info("CUDA not detected, installing CPU version of PyTorch and facenet_pytorch.")
    subprocess.check_call([
      sys.executable, "-m", "pip", "install",
      "torch==2.2.2",
      "torchvision==0.17.2",
      "torchaudio==2.2.2",
    ])
    subprocess.check_call([sys.executable,"-m",'pip','install','facenet_pytorch','--no-deps'])

class CustomInstallCommand(_install):
  """Customized setuptools install command to install the correct PyTorch version based on CUDA version."""
  def run(self):
    # Detect CUDA version
    print("Looking for installed CUDA...")
    cuda_version = get_cuda_version()
    if cuda_version is not None:
      print(f"Found existing CUDA:{cuda_version}")
      print("Installing torch,torchvision,torchaudio,facenet_pytorch,xformers...")
    else:
      print("No CUDA found - torch computations will be slow!")
      print("Install the appropiate version of torch and facenet_pytorch for your system after install!")
      print("Installing torch,torchvision,torchaudio...")

    # Install PyTorch,facenet_pytorch and xformers based on CUDA version
    try:
      install_pytorch(cuda_version)
      print("Install successful!")
    except Exception as e:
      print(f"Install failed with error: {str(e)}")
      print("To use the package please install the appropriate versions of the following libraries for your system:")
      print("torch==2.2.0\ntorchvision==0.17.2\ntorchaudio==2.2.2\nfacenet_pytorch==2.6.0")


    # Run the standard install process
    print("Installing pip dependencies...")
    _install.run(self)

setup(
  name="charloratools",
  version="0.2.3",
  description="Python package including tools that facilitate training images scraping, management, and filtering for stable diffusion character LoRa training.",
  author="svdC1",
  author_email="svdc1mail@gmail.com",
  maintainer="svdC1",
  maintainer_email="svdc1mail@gmail.com",
  url="https://github.com/svdC1/charloratools",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  python_requires=">=3.10.6",
  extras_require={
    'dev': ["coverage>=7.0.0,<8.0.0",
		"codecov>=2.0.0,<3.0.0",
		"jupyter>=1.0.0"
    ]
  },
  cmdclass={
    'install': CustomInstallCommand,
  },
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
)
