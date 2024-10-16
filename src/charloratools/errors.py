"""
> Defines **custom exception classes used throughout the application
to handle various types of errors**

Error Categories
----------------
 - Image Errors
 - General Errors
 - Selenium Errors
 - Shared Errors
 - Torch Errors

Examples
--------
```python
from charloratools.errors import ImageIsUnopenableError
try:
    # some code that may fail
except ImageIsUnopenableError as e:
    print(f"Error: {e}")
```

Classes
-------
ImageIsUnopenableError
    Exception raised when an image cannot be opened, indicating issues
    like corruption or incorrect format.
ImageTypeNotSupportedError
    Exception raised for unsupported image types, indicating that the
    format is not recognized by the application.
ImageIsDeletedError
    Exception raised when an operation is attempted on an image that
    has been marked as deleted.
ImgDeleteError
    Exception raised when deleting an image fails due to
    system errors or permission issues.
ImgOperationError
    Exception raised for general errors during image operations that are
    not covered by more specific exceptions.
ImgHashNotSupportedError
    Exception raised for unsupported image hashing methods,
    indicating a failure to hash images correctly.
InvalidTypeError
    Exception raised for invalid types encountered in the application,
    indicating a type mismatch.
OutOfRangeError
    Exception raised when a value is outside the acceptable range,
    such as exceeding defined limits.
InvalidInputError
    Exception raised for invalid user inputs, such as missing or
    incorrectly formatted parameters.
InvalidPathError
    Exception raised when a specified path does not exist or is invalid.
FailedToAddOptionsArgumentError
    Exception raised when failing to add arguments to the WebDriver
    during Selenium configuration.
DriverInitializationError
    Exception raised when the Selenium WebDriver fails to initialize,
    often due to environment issues.
ErrorScrollingPage
    Exception raised when an error occurs while scrolling
    a webpage with Selenium.
VSCOSignInError
    Exception raised for errors during signing into VSCO.
XSignInError
    Exception raised for errors during signing into X.
InstaSignInError
    Exception raised for errors during signing into Instagram.
NoImagesFoundInGalleryError
    Exception raised when no images are found in a specified gallery
    during operations.
NoImagesFoundError
    Exception raised when no images are found during an expected operation.
UsernameNotFoundError
    Exception raised when a specified username cannot be located in
    relevant contexts.
ImageDownloadError
    Exception raised for errors during the image downloading process
    due to various issues.
NoImagesInDirectoryError
    Exception raised when no images are found in the specified directory.
NoFaceDetectedInReferenceImage
    Exception raised when no face is detected in the provided reference image.
InfoDictFormatError
    Exception raised for format errors in the info dictionary returned
    by FaceRecognizer filtering methods.
FileOrDirPermissionDeniedError
    Exception raised when permission is denied for file or
    directory operations.
OperationNotSupportedError
    Exception raised when an attempted operation is not supported in
    the current context.
OperationResultsInEmptyDirectoryError
    Exception raised when an operation would result in an empty directory.
TorchNotInstalledError
    Exception raised when the required PyTorch library is not found
    in the environment.
"""


class ImageIsUnopenableError(Exception):
    """
    Exception raised when an image cannot be opened.

    This error is raised for issues related to opening an image file,
    such as corruption or incorrect format.
    """


class ImageTypeNotSupportedError(Exception):
    """
    Exception raised when an unsupported image type is encountered.

    This error indicates that the image file format is not supported
    by the application.
    """


class ImageIsDeletedError(Exception):
    """
    Exception raised when an operation is attempted on a deleted image.

    This error indicates that the image has been marked as deleted
    and cannot be processed.
    """


class ImgDeleteError(Exception):
    """
    Exception raised when there is an error during image deletion.

    This error is raised if an operation to delete an image fails
    due to an underlying issue (like file access permissions).
    """


class ImgOperationError(Exception):
    """
    Exception raised for general image operation errors.

    This error indicates a failure in performing an operation on an image
    that is not covered by more specific exceptions.
    """


class ImgHashNotSupportedError(Exception):
    """
    Exception raised when an unsupported image hashing method is requested.

    This error indicates a failure to use the specified hashing method
    for image comparison.
    """


class InvalidTypeError(Exception):
    """
    Exception raised for an invalid type encountered in the application.

    This error indicates that a value does not match the expected type.
    """


class OutOfRangeError(Exception):
    """
    Exception raised when a value is out of the expected range.

    This error is raised when a parameter value exceeds or falls below
    defined boundaries.
    """


class InvalidInputError(Exception):
    """
    Exception raised for invalid user inputs.

    This error indicates that the input provided does not meet
    the validation requirements.
    """


class InvalidPathError(Exception):
    """
    Exception raised when a specified path is invalid.

    This error indicates that the provided path does not exist
    or is not accessible.
    """


class FailedToAddOptionsArgumentError(Exception):
    """
    Exception raised when a failure occurs while adding options
    to the WebDriver.

    This error flags problems encountered when configuring the
    Selenium WebDriver, such as invalid option arguments.
    """


class DriverInitializationError(Exception):
    """
    Exception raised when the Selenium WebDriver fails to initialize.

    This error indicates that the driver could not be created,
    often due to environment issues or configuration errors.
    """


class ErrorScrollingPage(Exception):
    """
    Exception raised when an error occurs while scrolling a webpage.

    This error indicates issues encountered during the scrolling
    process with the Selenium WebDriver.
    """


class VSCOSignInError(Exception):
    """
    Exception raised for errors during signing into VSCO.

    This error indicates that a sign-in attempt to VSCO has failed.
    """


class XSignInError(Exception):
    """
    Exception raised for errors during signing into X.

    This error indicates that a sign-in attempt to X has failed.
    """


class InstaSignInError(Exception):
    """
    Exception raised for errors during signing into Instagram.

    This error indicates that a sign-in attempt to Instagram has failed.
    """


class NoImagesFoundInGalleryError(Exception):
    """
    Exception raised when no images are found in the gallery.

    This error indicates that a gallery operation is attempted but there
    are no images available.
    """


class NoImagesFoundError(Exception):
    """
    Exception raised when no images are found during an operation.

    This error indicates that an operation expected images but
    found none.
    """


class UsernameNotFoundError(Exception):
    """
    Exception raised when a specified username cannot be found.

    This error indicates that an operation was unable to locate the
    specified username in relevant contexts, such as scraping.
    """


class ImageDownloadError(Exception):
    """
    Exception raised when an error occurs during image downloading.

    This error indicates that there was a failure in retrieving an
    image from a specified source.
    """


class NoImagesInDirectoryError(Exception):
    """
    Exception raised when no images are found in a specified directory.

    This error indicates that an operation that required images
    found none in the given directory.
    """


class NoFaceDetectedInReferenceImage(Exception):
    """
    Exception raised when no face is detected in the reference image.

    This error indicates that a required face for processing was not found
    in the specified image.
    """


class InfoDictFormatError(Exception):
    """
    Exception raised for format errors in the info dictionary.

    This error indicates that the provided dictionary does not conform
    to the expected structure or content.
    """


class FileOrDirPermissionDeniedError(Exception):
    """
    Exception raised when file or directory access is denied.

    This error indicates issues with permissions preventing file or
    directory operations from succeeding.
    """


class OperationNotSupportedError(Exception):
    """
    Exception raised when an attempted operation is unsupported.

    This error indicates that the functionality requested cannot be
    fulfilled based on the current context or parameters.
    """


class OperationResultsInEmptyDirectoryError(Exception):
    """
    Exception raised when an operation would result in an empty directory.

    This error indicates that the requested operation would lead to
    no files being available wherever applicable.
    """


class TorchNotInstalledError(Exception):
    """
    Exception raised when the required PyTorch library is not installed.

    This error indicates that PyTorch is necessary for operations
    but cannot be found in the environment.
    """
