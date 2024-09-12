import os
import platform
import subprocess
from setuptools import setup,find_packages  
from setuptools.command.install import install
import logging
import re
import sys
import ensurepip 

#Ensure pip is installed
try:
  import pip
except ImportError:
  ensurepip.bootstrap()
  
#Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InstallCommand(install):
  """
  Checks the operating system details and installs the correct distributions for torch,torchvision,torchaudio and facenet_pytorch 
  """
  
  def run(self):
    #Installing torch,torchvision and torchaudio first
    self.install_pytorch()
    #Installing facenet_pytorch
    self.install_facenet_pytorch()
    #Proceed with normal installation
    install.run(self)
  
  def run_os_command(self,command):
    """
    Helper function to run system commands and manage error handling
    """
    try:
      result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      logging.info(result.stdout.decode('utf-8'))
      return result.stdout.decode('utf-8')
    
    except subprocess.CalledProcessError as e:
      logging.error(f"Command failed: {command}")
      logging.error(e.stderr.decode('utf-8'))
      logging.error(f"Command failed: {command}")
      logging.error(f"stdout: {e.stdout.decode('utf-8')}")
      logging.error(f"stderr: {e.stderr.decode('utf-8')}")
      raise RuntimeError(f"Command {command} failed with error: {e.stderr.decode('utf-8')}")
  
  def get_cuda_version(self):
    """
    Checks if CUDA is available, if it is , return the CUDA version installed
    """
    try:
      # Check if an NVIDIA GPU is present
      subprocess.run("nvidia-smi", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      logging.info("NVIDIA GPU detected.")
      # Check the installed CUDA version
      nvcc_output = self.run_os_command("nvcc --version")
      cuda_version_match = re.search(r"release (\d+\.\d+)", nvcc_output)
      if cuda_version_match:
        cuda_version = cuda_version_match.group(1)
        logging.info(f"Detected CUDA version: {cuda_version}")
        return cuda_version
      else:
        logging.info("Unable to detect CUDA version from nvcc output.")
        return None
      
    except subprocess.CalledProcessError:
      logging.info("CUDA is not available.")
      return None
  
  def check_rocm(self):
    """Check if ROCm (AMD GPU) is available."""
    try:
      rocm_output = self.run_os_command("rocminfo")
      if "AMD" in rocm_output or "Radeon" in rocm_output:
        logging.info("AMD GPU detected (ROCm).")
        return True
      else:
        logging.info("No ROCm-compatible GPU detected.")
        return False
    except subprocess.CalledProcessError:
      logging.info("ROCm is not available.")
      return False
  
  def install_pytorch(self):
    """Install the appropriate version of Torch based on the platform and CUDA version."""
    
    os_name = platform.system()
    cuda_version = self.get_cuda_version()
    is_rocm = False
    #Checking if ROCm is available (Linux only)
    if os_name == 'Linux':
      is_rocm = self.check_rocm()
      if is_rocm:
        #Installation for Linux with ROCm installed
        logging.info(f"Found ROCm, installing Torch distribution for ROCm 5.7 support for Linux")
        self.run_os_command("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7")
        
    
    #Installation for Windows and Linux
    if os_name == 'Linux' or os_name == 'Windows':
      if cuda_version:
        logging.info(f"Found CUDA version {cuda_version}, checking support for {os_name}.")
        # Install Torch for CUDA 11.8
        if cuda_version=="11.8":
          logging.info(f"Trying to install Torch distribution for CUDA 11.8 support for {os_name}.")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118")
          
        #Install Torch for CUDA 12.1
        elif cuda_version=="12.1":
          logging.info(f"Trying to install Torch distribution for CUDA 12.1 support for {os_name}.")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121")
        
        #Install Torch for CPU Only
        else:
          logging.warning(f"Unsupported CUDA version {cuda_version}.Supported are : 11.8 and 12.1. Falling back to CPU-only Torch.")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
          
    #Installation on MacOs    
    elif os_name == 'Darwin':
      # macOS installation (CPU-only as macOS doesn't support CUDA or ROCm)
      logging.info("Detected macOs operating system, Trying to install Torch for CPU only")
      self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
        
    else:
      logging.error(f"Unsupported platform: {os_name}")
      raise RuntimeError(f"Unsupported platform: {os_name}")
  
  def install_facenet_pytorch(self):
    """
    Install the facenet_pytorch library after Torch has been installed.
    """
    # Install facenet_pytorch (ensure it works with the installed PyTorch version)
    logging.info("Installing facenet_pytorch>=2.6.0")
    self.run_os_command("pip install facenet-pytorch>=2.6.0")

setup(
  name="charloratools",
  version="0.2.5",
  description="Python package including tools that facilitate training images scraping, management, and filtering for stable diffusion character LoRa training.",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  python_requires=">=3.11",
  extras_require={
    'dev': ["coverage>=7.0.0,<8.0.0",
		"codecov>=2.0.0,<3.0.0",
    "pytest",
    "build",
    "twine"
    ]
  },
  cmdclass={
    'install': InstallCommand,
  }
)
