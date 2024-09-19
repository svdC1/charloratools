from setuptools import setup, find_packages

setup(
    name="charloratools",
    version="1.0.1",
    description="""Python package including tools that facilitate training
    images scraping, management, and filtering for stable diffusion
    character LoRa training.""",
    packages=find_packages(where="src", include=[
                           'charloratools',
                           'charloratools.*',
                           'charloratools/facenet_pytorch',
                           'charloratools/facenet_pytorch.*']),
    include_package_data=True,
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
        'full': ['torch', 'torchvision', 'torchaudio']
    }
)
