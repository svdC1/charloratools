from setuptools import setup,find_packages  

setup(
  name="charloratools",
  version="0.3.1",
  description="Python package including tools that facilitate training images scraping, management, and filtering for stable diffusion character LoRa training.",
  packages=find_packages(where="src"),
  package_dir={"": "src"},
  python_requires=">=3.11",
  extras_require={
    'dev': ["coverage>=7.0.0,<8.0.0",
		"codecov>=2.0.0,<3.0.0",
    "pytest",
    "build",
    "twine",
    "pytest-cov"
    ],
    'full':['torch==2.2.2', 'torchvision==0.17.2' ,'torchaudio==2.2.2','facenet_pytorch>=2.6.0']
  }
)
