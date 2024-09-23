import subprocess
import platform
import logging
import re
import argparse
import sys

EXECUTABLE = sys.executable


def run_os_command(command_args: list) -> subprocess.CompletedProcess:
    """
    Helper function to run system commands and manage error handling
    """
    logging.info(f"Running : {' '.join(command_args)}")
    try:
        # stdout:str; stderr:str; returncode:int;
        result = subprocess.run(
            command_args, capture_output=True, text=True,
            check=True, encoding='utf-8')
        logging.info(f"Command Return Code : {result.returncode}")
        return result

    except subprocess.CalledProcessError:
        logging.error(f"Command Failed To Run; Error:{result.stderr}")
        return result


def get_cuda_version() -> str | None:
    """
     Checks if CUDA is available, if it is , return the CUDA version installed
    """
    logging.info("Checking if CUDA is available...")
    try:
        nvidia_smi = run_os_command(['nvidia-smi'])
    except FileNotFoundError:
        logging.info("CUDA not available - File Not Found Error")
        return None

    if nvidia_smi.returncode != 0:
        logging.warning("CUDA is not available.")
        return None
    else:
        logging.info("Found CUDA, checking nvcc for version...")
        nvcc = run_os_command(['nvcc', '--version'])
        cuda_version_match = re.search(r"release (\d+\.\d+)", nvcc.stdout)
        if nvcc.returncode != 0:
            logging.warning("CUDA is not available")
        else:
            if not cuda_version_match:
                logging.warning(
                    "Couldn't detect CUDA version from nvcc output")
                return None
            else:
                cuda_version = cuda_version_match.group(1)
                logging.info(f"Detected CUDA version: {cuda_version}")
                return cuda_version


def check_rocm() -> bool | None:
    """Check if ROCm (AMD GPU) is available."""
    logging.info("Checking if ROCM is available...")
    rocm = run_os_command(["rocminfo"])
    if rocm.returncode != 0:
        logging.warning("ROCm is not available.")
        return False

    else:
        if "AMD" in rocm or "Radeon" in rocm:
            logging.info("AMD GPU detected (ROCm).")
            return True
        else:
            logging.info("No ROCm-compatible GPU detected.")
            return False


def get_os() -> str | None:
    if platform.system() == "":
        return None
    else:
        logging.info(f"Detected OS {platform.system()}")
        return platform.system()


def install_torch_cpu() -> None:
    """
    Torch CPU-Only installation
    """
    # pip install torch torchvision torchaudio --index-url
    # https://download.pytorch.org/whl/cpu
    logging.warning("Falling back to CPU-Only Torch install")
    cpu_args = [EXECUTABLE, "-m", "pip", "install", "torch", "torchvision",
                "torchaudio", "--index-url",
                "https://download.pytorch.org/whl/cpu"]
    result = run_os_command(cpu_args)
    if result.returncode != 0:
        raise RuntimeError(
            f"Error while installing Torch CPU-Only; Error: {result.stderr}")
    else:
        print(result.stdout)


def install_torch_cuda(cuda_version: str | None) -> None:
    """
    Torch installation with CUDA Support
    """
    if cuda_version is None:
        logging.warning("CUDA is not installed, installing CPU-Only torch")
        install_torch_cpu()

    elif cuda_version.startswith("11"):
        # pip install torch torchvision torchaudio --index-url
        # https://download.pytorch.org/whl/cu118
        logging.info("Installing Torch for CUDA 11")
        cu11_args = [EXECUTABLE, "-m", "pip", "install", "torch",
                     "torchvision",
                     "torchaudio", "--index-url",
                     "https://download.pytorch.org/whl/cu118"]
        result = run_os_command(cu11_args)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error installing Torch with CUDA 11 : {result.stderr}")
        else:
            print(result.stdout)

    elif cuda_version.startswith("12.4"):
        logging.info("Installing Torch for CUDA 12.4")
        # pip install torch torchvision torchaudio --index-url
        # https://download.pytorch.org/whl/cu124
        cu124_args = [EXECUTABLE, "-m", "pip", "install", "torch",
                      "torchvision",
                      "torchaudio", "--index-url",
                      "https://download.pytorch.org/whl/cu124"]
        result = run_os_command(cu124_args)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error while installing Torch with CUDA 12 : {result.stderr}")
        else:
            print(result.stdout)

    elif cuda_version.startswith("12"):
        logging.info("Installing Torch for CUDA 12.1")
        # pip install torch torchvision torchaudio --index-url
        # https://download.pytorch.org/whl/cu121
        cu12_args = [EXECUTABLE, "-m", "pip", "install", "torch",
                     "torchvision",
                     "torchaudio", "--index-url",
                     "https://download.pytorch.org/whl/cu121"]
        result = run_os_command(cu12_args)
        if result.returncode != 0:
            raise RuntimeError(
                f"Error while installing Torch with CUDA 12 : {result.stderr}")
        else:
            print(result.stdout)

    else:
        logging.warning(
            f"Unsupported CUDA -v{cuda_version}.Supported are : 11.x and 12.x")
        install_torch_cpu()


def install_torch_rocm() -> None:
    """
    Torch installation with ROCM support - Linux Only
    """
    # pip install torch torchvision torchaudio --index-url
    # https://download.pytorch.org/whl/rocm6.1
    logging.info(
        "Found ROCm, installing Torch distribution - ROCm 5.7 for Linux")
    rocm_args = [EXECUTABLE, "-m", "pip", "install", "torch", "torchvision",
                 "torchaudio", "--index-url",
                 "https://download.pytorch.org/whl/rocm6.1"]
    result = run_os_command(rocm_args)
    if result.returncode != 0:
        raise RuntimeError(
            f"Error while installing Torch with ROCm : {result.stderr}")
    else:
        print(result.stdout)


def get_torch_version() -> str | None:
    try:
        import torch
        t_version = torch.__version__
        dist = ""
        logging.info(f"Found Torch version {t_version}")
        # Checking for different distributions
        if "+" in t_version:
            dist = t_version.split("+")[1]
            t_version = t_version.split("+")[0]

        return (t_version, dist)

    except ImportError:
        logging.warning("No Torch Installation found")
        return None


def run_install_script() -> None:
    # Checking system details
    os = get_os()
    if not os:
        raise RuntimeError("Couldn't determine OS.")

    check_torch = get_torch_version()
    if check_torch is not None:
        t_version = check_torch[0]
        t_dist = check_torch[1]

    cuda_version = get_cuda_version()
    try:
        rocm_found = check_rocm()
    except FileNotFoundError:
        logging.warning("No ROCm found - (Windows File Not Found Error)")
        rocm_found = False

    if check_torch is not None:
        logging.info(
            f"Found torch ({t_version} + distribution {t_dist}) - Exiting")
        return None

    else:
        # CPU-Only Installation for MacOS
        if os == "Darwin":
            install_torch_cpu()
            logging.info("Installation Completed")

        # ROCm Linux Installation
        elif os == "Linux" and rocm_found:
            install_torch_rocm()
            logging.info("Installation Completed")

        # Windows or Linux Installation (Checking for CUDA)
        elif (os == "Linux" or os == "Windows"):
            install_torch_cuda(cuda_version)
            logging.info("Installation Completed")

        # Windows with ROCm - No Torch support
        else:
            install_torch_cpu()
            logging.info("Installation Completed")


def main():
    # Setup logging
    logging.basicConfig(
        level="INFO", style='{', format="{levelname} - {message}")
    parser = argparse.ArgumentParser(
        prog="charloratools", description="charloratools CLI")
    # Create sub-parsers
    subparsers = parser.add_subparsers(
        dest='command', help="Available Commands: ")

    subparsers.add_parser(
        'install_torch',
        help="Check system specs and install correct torch version.")

    args = parser.parse_args()

    if args.command == "install_torch":
        run_install_script()


if __name__ == "__main__":
    main()
