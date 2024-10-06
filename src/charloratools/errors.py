# -------Image Errors--------
class ImageIsUnopenableError(Exception):
    pass


class ImageTypeNotSupportedError(Exception):
    pass


class ImageIsDeletedError(Exception):
    pass


class ImgDeleteError(Exception):
    pass


class ImgOperationError(Exception):
    pass


class ImgHashNotSupportedError(Exception):
    pass
# ---------------------------
# ------General Errors-------
# Global Errors


class InvalidTypeError(Exception):
    pass


class OutOfRangeError(Exception):
    pass


class InvalidInputError(Exception):
    pass


class InvalidPathError(Exception):
    pass
# -------------------------------
# -----Selenium Errors-----------
# ------Webdriver-------


class FailedToAddOptionsArgumentError(Exception):
    pass


class DriverInitializationError(Exception):
    pass


class ErrorScrollingPage(Exception):
    pass
# ----VSCO Scraper


class VSCOSignInError(Exception):
    pass
# ----X Scraper


class XSignInError(Exception):
    pass
# ----Instagram Scraper


class InstaSignInError(Exception):
    pass
# -----Shared


class NoImagesFoundInGalleryError(Exception):
    pass


class NoImagesFoundError(Exception):
    pass


class UsernameNotFoundError(Exception):
    pass


class ImageDownloadError(Exception):
    pass
# -------------System Directory Errors------


class NoImagesInDirectoryError(Exception):
    pass


class NoFaceDetectedInReferenceImage(Exception):
    pass


class InfoDictFormatError(Exception):
    pass


class FileOrDirPermissionDeniedError(Exception):
    pass


class OperationNotSupportedError(Exception):
    pass


class OperationResultsInEmptyDirectoryError(Exception):
    pass
# -------------Torch Errors-----------------------------


class TorchNotInstalledError(Exception):
    pass
