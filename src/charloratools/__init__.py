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
            else:
                raise
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
