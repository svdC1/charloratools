from setuptools import setup, find_packages

setup(
    name="charloratools",
    version="1.0.6",
    description="""Python package including tools that facilitate training
    images scraping, management, and filtering for stable diffusion
    character LoRa training.""",
    packages=find_packages(where="src", include=[
                           'charloratools',
                           'charloratools.*',
                           'tests/conftest.py'
                           'charloratools/facenet_pytorch',
                           'charloratools/facenet_pytorch.*',
                           'charloratools/facenet_pytorch/data/*']),
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
        'cpu': ['torch', 'torchvision', 'torchaudio'],
        'docs': ["black", "mkdocs",
                 "mkdocstrings",
                 "mkdocstrings-python",
                 "mkdocs-autorefs",
                 "mkdocs-get-deps"
                 "mkdocs-material>9.5.0",
                 "mkdocs-material-extensions",
                 "markdown",
                 "pymdown-extensions",
                 "Pygments"]
    }
)
