import subprocess
import importlib.util
import importlib.metadata
import platform
import logging
import re
import argparse


class CustomTorchInstall:
  """
  Checks the operating system details and installs the correct distributions for torch,torchvision,torchaudio and facenet_pytorch 
  """
  
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
        print("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7")
        self.run_os_command("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7")
        
    
    #Installation for Windows and Linux
    if os_name == 'Linux' or os_name == 'Windows':
      if cuda_version:
        logging.info(f"Found CUDA version {cuda_version}, checking support for {os_name}.")
        # Install Torch for CUDA 11.8
        if cuda_version.startswith("11"):
          logging.info(f"Trying to install Torch distribution for CUDA 11.8 support for {os_name}.")
          print("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118")
          
        #Install Torch for CUDA 12.1
        elif cuda_version.startswith("12"):
          logging.info(f"Trying to install Torch distribution for CUDA 12.1 support for {os_name}.")
          print("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121")
        
        #Install Torch for CPU Only
        else:
          logging.warning(f"Unsupported CUDA version {cuda_version}.Supported are : 11.8 and 12.1. Falling back to CPU-only Torch.")
          print("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
          self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
          
    #Installation on MacOs    
    elif os_name == 'Darwin':
      # macOS installation (CPU-only as macOS doesn't support CUDA or ROCm)
      logging.info("Detected macOs operating system, Trying to install Torch for CPU only")
      print("pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
      self.run_os_command(f"pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu")
        
    else:
      logging.error(f"Unsupported platform: {os_name}")
      print(f"Unsupported platform: {os_name}")
      raise RuntimeError(f"Unsupported platform: {os_name}")
  
  def install_facenet_pytorch(self):
    """
    Install the facenet_pytorch library after Torch has been installed.
    """
    # Install facenet_pytorch (ensure it works with the installed PyTorch version)
    logging.info("Installing facenet_pytorch>=2.6.0")
    print("pip install facenet-pytorch>=2.6.0")
    self.run_os_command("pip install facenet-pytorch>=2.6.0")


def install_torch():
  #Setup logging
  logging.basicConfig(level="INFO",style='{', format="{levelname} - {message}")
  #Checking if torch is installed
  torch_spec = importlib.util.find_spec("torch")
  torchvision_spec=importlib.util.find_spec("torchvision")
  torchaudio_spec=importlib.util.find_spec("torchaudio")
  facenet_pytorch_spec= importlib.util.find_spec("facenet_pytorch")
  cti=CustomTorchInstall()
  if torch_spec is None:
    print("No torch installation found,installing...")
    cti.install_pytorch()
    
  else:
    import torch
    version= torch.__version__
    if '+' in version:
      version=version.split("+")[0]
    print(f"Torch version installed: {version}")
    if version!="2.2.2":
      print("Torch version is incompatible, installing 2.2.2 ...")
      cti.install_pytorch()
    else:
      print("Torch version matches,checking torchvision and torchaudio")
      if not torchvision_spec or not torchaudio_spec:
        print("No torchvision or torchaudio found,reinstalling torch...")
        cti.install_pytorch()
      else:
        import torchvision
        import torchaudio
        
        vision_version=torchvision.__version__
        audio_version=torchaudio.__version__
        if '+' in vision_version:
          vision_version=vision_version.split("+")[0]
        if '+' in audio_version:
          audio_version=audio_version.split("+")[0]
          
        if vision_version!="0.17.2":
          print(f"Incompatible torchvision version {vision_version} found,reinstalling torch...")
          cti.install_pytorch()
        
        elif audio_version!="2.2.2":
          print(f"Incompatible torchaudio version {audio_version} found,reinstalling torch...")
          cti.install_pytorch()
        else:
          print("torchvision and torchaudio are correctly installed.")
  
  print("Finished checking torch.")
  if facenet_pytorch_spec is None:
    print("No facenet-pytorch installation found,installing...")
    cti.install_facenet_pytorch()
  else:
    import facenet_pytorch
    fp_version=importlib.metadata.version('facenet-pytorch')
    if fp_version!='2.6.0':
      print(f"Incompatible facenet-pytorch {fp_version},reinstalling...")
      cti.install_facenet_pytorch()
    else:
      print("facenet-pytorch is correctly installed.")
  
  print("torch,torchvision,torchaudio,facenet-pytorch are correctly installed")

def main():
  parser = argparse.ArgumentParser(prog="charloratools")
  subparsers = parser.add_subparsers(dest="command")
  # Create install_torch subcommand
  parser_install = subparsers.add_parser('install_torch', help='Custom Torch and Facenet-Pytorch installation script')
  args = parser.parse_args()
  if args.command == "install_torch":
    install_torch()
    
if __name__=="__main__":
  main()
  
        