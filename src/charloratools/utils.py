"""
Functionalities
---------------
 - image processing
 - facial recognition
 - scraping functionalities
 - validating image directories
 - downloading from URLs
 - initializing Selenium drivers
 - various image transformation utilities.

Functions
---------
torch_import()
    Attempts to import the PyTorch library and its torchvision transforms.
    Raises errors if unsuccessful.
dirisvalid(path, check_images=True, return_info=False, hashtype=None,
           create_if_not_found=False, show_tqdm=False)
    Validates a directory path and checks for image files.
    Optionally creates the directory if it does not exist.
GetUniqueDtStr()
    Generates a unique timestamp string for file naming.
save_with_detection_box(img_path, outdir, boxes)
    Saves an image with drawn detection boxes to a specified output directory.
distance_function(embedding1, embedding2, method, classify=False,
                  threshold=None)
    Calculates the distance between two embeddings using either
    Euclidean distance or cosine similarity.
InfoDict2Pandas(info)
    Converts and validates an info dictionary returned by filtering
    methods into a formatted Pandas DataFrame.
is_matched(df_el)
    Returns the DataFrame element if it matches (i.e., 'matched' is True).
not_is_matched(df_el)
    Returns the DataFrame element if it does not match
    (i.e., 'matched' is False).
split_matched(info_dict)
    Separates an info dictionary into matched and not matched items,
    returning Pandas DataFrames.
initialize_driver(headless=True, incognito=True, add_arguments=None)
    Initializes a Selenium Chrome Web Driver with specified options.
page_scroll(driver, n, webpage_wait_time)
    Scrolls a webpage a specified number of times using Selenium WebDriver.
download_from_src(srcs, prefix, save_path, logger)
    Downloads images from provided URL sources and checks for corruption.
img_path_to_tensor(img_path, nsize=None)
    Converts an image at the specified path to a tensor,optionally resizing it.
dir_path_to_img_batch(path)
    Converts all images in a directory to a batch of tensors.

Raises
------
TorchNotInstalledError
    Raised when the required PyTorch library is not found.
InvalidInputError
    Raised for invalid inputs or parameters.
InvalidTypeError
    Raised when the type of the provided path is not suitable.
NoImagesInDirectoryError
    Raised when no images are found in the specified directory.
ErrorScrollingPage
    Raised when an error occurs while scrolling a webpage with Selenium.
FailedToAddOptionsArgumentError
    Raised when additional arguments for the webdriver cannot be added.
DriverInitializationError
    Raised when the Selenium driver fails to initialize.
ImageDownloadError
    Raised when an error occurs during image downloading.
InfoDictFormatError
    Raised when the format of the info dictionary is invalid.

Examples
--------
```python
from charloratools.utils import dirisvalid, download_from_src
valid_dir, image_info = dirisvalid('path/to/images', return_info=True)
download_from_src(['http://example.com/image1.jpg'],
                   'prefix', 'path/to/download', logger)
```
"""

import logging
import time
import requests
from pathlib import Path
from datetime import datetime
# External Libs
import PIL
from PIL import Image, ImageDraw
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm, trange
# Scripts
from . import errors
from . import SysFileManager


def torch_import():
    """
    Attempts to import the PyTorch library and its torchvision transforms.

    Returns
    -------
    tuple
        A tuple containing the PyTorch and torchvision transforms modules.

    Raises
    ------
    TorchNotInstalledError
        If the PyTorch library is not installed.
    """
    try:
        import torch
        from torchvision.transforms import v2
        return torch, v2
    except ImportError:
        raise errors.TorchNotInstalledError


def dirisvalid(path: str | Path,
               check_images: bool = True,
               return_info: bool = False,
               hashtype: str | None = None,
               create_if_not_found: bool = False,
               show_tqdm: bool = False):
    """
    Checks the validity of a directory and its image contents or
    creates a new directory.

    Parameters
    ----------
    path : str or Path
        Path to the directory to validate or create.
    check_images : bool, optional
        Whether to check if at least one image exists in the directory.
        Defaults to True.
    return_info : bool, optional
        If True, returns number of images in the directory and their paths.
        Defaults to False.
    hashtype : str or None, optional
        Hash type to use for hashing images.
        Must be provided if return_info=True. Defaults to None.
    create_if_not_found : bool, optional
        If True, creates the directory if it does not exist. Defaults to False.
    show_tqdm : bool, optional
        If True, shows progress with tqdm during image loading and hashing.
        Defaults to False.

    Returns
    -------
    Path or tuple
        If return_info is False, returns the resolved directory path.
        If return_info is True, returns a tuple containing the directory
        path and a dictionary of images.

    Raises
    ------
    InvalidTypeError
        If the provided path is not a string or path-like object.
    InvalidPathError
        If the directory does not exist and create_if_not_found is False.
    NoImagesInDirectoryError
        If no images are found in the directory when check_images is True.
    """
    torch, v2 = torch_import()
    logger = logging.getLogger(__name__)
    if return_info and hashtype is None:
        raise errors.InvalidInputError(
            "hashtype must be provided if return_info=True")

    if isinstance(path, str):
        path = Path(path).resolve()
    elif isinstance(path, Path):
        path = path.resolve()
    else:
        raise errors.InvalidTypeError(
            "Path must be a string or path-like object")
    # Create Directory and send path if it isn't found and arg is set to true
    if not path.exists():
        if create_if_not_found:
            path.mkdir(parents=True, exist_ok=True)
            path = path.resolve()
            logger.debug(f'Created empty directory {path.resolve()}')
            return path
        else:
            raise errors.InvalidPathError("Path does not exist")
    # If arg is set to true and dir exists return path
    else:
        if create_if_not_found:
            return path.resolve()
    # If not img arg is set to False validate
    if not check_images:
        return True
    # Checking Images
    img_paths = [path/i for i in path.iterdir() if i.suffix.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.tiff', '.ppm', '.bmp', '.gif'))]
    if len(img_paths) == 0:
        raise errors.NoImagesInDirectoryError("No images found in directory")
    # Checking if pillow can open the image,else deleting
    images = {}
    if show_tqdm:
        with logging_redirect_tqdm():
            for img_path in tqdm(img_paths, desc='Checking and Hashing Imgs'):
                try:
                    with Image.open(img_path) as img:
                        # Checking if operation can be performed on img
                        np.array(img)
                except PIL.UnidentifiedImageError:
                    img_path.unlink()
                    logger.debug(f'Removed {img_path} due to PIL error')

                images[img_path] = SysFileManager.ImgManager(
                    img_path, hashtype=hashtype)
    else:
        for img_path in img_paths:
            try:
                with Image.open(img_path) as img:
                    # Checking if operation can be performed on img
                    np.array(img)
            except PIL.UnidentifiedImageError:
                img_path.unlink()
                logger.debug(f'Removed {img_path} due to PIL error')

            images[img_path] = SysFileManager.ImgManager(
                img_path, hashtype=hashtype)

    if return_info:
        logger.debug("Directory is Valid")
        logger.debug(f"Found {len(images)} images in directory")
        return path, images
    else:
        logger.debug("Directory is Valid")
        return True


def GetUniqueDtStr():
    """
    Generates a unique timestamp string for file naming.

    Returns
    -------
    str
        A unique timestamp string formatted as `MMDDYYHHMMSSffffff`.
    """
    return datetime.now().strftime("%m%d%y%H%M%S%f")

# ---------Face Recognizer Utils------------


def save_with_detection_box(img_path: str | Path, outdir: str | Path, boxes):
    """
    Saves an image with detection boxes drawn on it to the specified
    output directory.

    Parameters
    ----------
    img_path : str or Path
        Path to the image file.
    outdir : str or Path
        Path to the directory where the output image will be saved.
    boxes : list
        List of bounding boxes to be drawn on the image.

    Raises
    ------
    InvalidTypeError
        If img_path or outdir is not of string or Path type.
    """
    torch, v2 = torch_import()
    outdir = dirisvalid(outdir, create_if_not_found=True)
    img_manager = SysFileManager.ImgManager(path=img_path)
    with Image.open(img_manager.path) as img:
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box.tolist(), outline='red', width=5)
        if img_manager.ext == ".jpg":
            img.save(outdir/img_manager.basename, quality=95)
        else:
            img.save(outdir/img_path.basename)


def distance_function(embedding1,
                      embedding2,
                      method,
                      classify=False,
                      threshold=None):
    """
    Calculates the distance between two embeddings using a specified method.

    Parameters
    ----------
    embedding1 : torch.Tensor
        The first embedding tensor.
    embedding2 : torch.Tensor
        The second embedding tensor.
    method : str
        The distance method to use ('euclidean' or 'cosine').
    classify : bool, optional
        If True, classify embeddings as matching based on the provided
        threshold. Defaults to False.
    threshold : float, optional
        Threshold for classifying embeddings when classify is True.
        Defaults to None.

    Returns
    -------
    float or bool
        If classify is False, returns the calculated distance.
        If classify is True, returns whether
        the embeddings match based on the threshold.

    Raises
    ------
    InvalidInputError
        If the provided method is not supported or if threshold is None
        when classify is True.
    """
    torch, v2 = torch_import()
    supported_methods = ['euclidean', 'cosine']
    if method not in supported_methods:
        raise errors.InvalidInputError('Method not supported')
    if classify and threshold is None:
        raise errors.InvalidInputError(
            'Threshold must be provided when classify is set to True')
    if method == 'euclidean':
        euclidean_distance = (embedding1 - embedding2).norm()
        if classify:
            if torch.any(euclidean_distance >= threshold):
                # Doesn't match
                return False
            else:
                return True
        else:
            return euclidean_distance
    elif method == 'cosine':
        cosine_similarity = torch.nn.functional.cosine_similarity(
            embedding1, embedding2)
        if classify:
            if torch.any(cosine_similarity >= threshold):
                # Matches
                return True
            else:
                return False
        else:
            return cosine_similarity


def InfoDict2Pandas(info: dict | list):
    """
    Validates and converts an info dictionary to a Pandas DataFrame.

    Parameters
    ----------
    info : dict or list
        Info dictionary returned by filtering methods of FaceRecognizer.

    Returns
    -------
    dict
        A dictionary containing DataFrames of matched and not matched items.

    Raises
    ------
    InfoDictFormatError
        If the format of the provided info dictionary is invalid.
    """
    torch, v2 = torch_import()
    if not isinstance(info, list) and not isinstance(info, dict):
        raise errors.InfoDictFormatError('Must be a dict or list of dicts')
    possible_outer_keys = ['imgs_with_face', 'imgs_without_face',
                           'imgs_with_ref_face',
                           'imgs_without_ref_face', 'imgs_with_multiple_faces',
                           'imgs_with_one_face']
    expected_inner_keys = ['path', 'boxes', 'probs', 'matched']
    # Validating Input
    if isinstance(info, dict):
        if 'info_dict_lst' in info.keys():
            if not isinstance(info['info_dict_lst'], list):
                raise errors.InfoDictFormatError(
                    'info_dict_lst Must be a list of dicts')
            for d in info['info_dict_lst']:
                if not isinstance(d, dict):
                    raise errors.InfoDictFormatError(
                        'info_dict_lst must be a list of dicts')
                for key in expected_inner_keys:
                    if key not in d.keys():
                        raise errors.InfoDictFormatError(
                            'One of the expected keys was not found')
                if 'distance' in d.keys():
                    if isinstance(d['distance'], torch.Tensor):
                        d['distance'] = d['distance'].detach().cpu().numpy()
        else:
            raise errors.InfoDictFormatError('Did not find info_dict_lst key')
    else:
        for d in info:
            if not isinstance(d, dict):
                raise errors.InfoDictFormatError(
                    'info_dict_lst must be a list of dicts')
            for key in expected_inner_keys:
                if key not in d.keys():
                    raise errors.InfoDictFormatError(
                        'One of the expected keys of was not found')
            if 'distance' in d.keys():
                if isinstance(d['distance'], torch.Tensor):
                    d['distance'] = d['distance'].detach().cpu().numpy()
    # ----
    # ---- Normalizing Boxes None format
    if isinstance(info, dict):
        for d in info['info_dict_lst']:
            if isinstance(d['boxes'], list):
                d['boxes'] = np.array(d['boxes'])
            if isinstance(d['boxes'], np.ndarray):
                if np.any(d['boxes'] is None):
                    d['boxes'] = None
    else:
        for d in info:
            if isinstance(d['boxes'], list):
                d['boxes'] = np.array(d['boxes'])
            if isinstance(d['boxes'], np.ndarray):
                if np.any(d['boxes'] is None):
                    d['boxes'] = None
    # ---- Sorting info
    fdict = {}
    fdistance_dict = {}
    pkeys_present = {}
    if isinstance(info, dict):
        for d in info['info_dict_lst']:
            if 'distance' in d.keys():
                for k, v in d.items():
                    if k not in fdistance_dict.keys():
                        fdistance_dict[k] = [v]
                    else:
                        fdistance_dict[k].append(v)
            else:
                for k, v in d.items():
                    if k not in fdict.keys():
                        fdict[k] = [v]
                    else:
                        fdict[k].append(v)

        for k, v in info.items():
            if k in possible_outer_keys:
                pkeys_present[k] = v
    else:
        for d in info:
            if 'distance' in d.keys():
                for k, v in d.items():
                    if k not in fdistance_dict.keys():
                        fdistance_dict[k] = [v]
                    else:
                        fdistance_dict[k].append(v)
            else:
                for k, v in d.items():
                    if k not in fdict.keys():
                        fdict[k] = [v]
                    else:
                        fdict[k].append(v)
    # ---- Converting to Pandas
    out = {}
    if len(fdistance_dict) > 0:
        out['matched_ref_df'] = pd.DataFrame(fdistance_dict)
    if len(pkeys_present) > 0:
        out['n_filtered_series'] = pd.Series(pkeys_present)
    out['info_df'] = pd.DataFrame(fdict)
    return out


def is_matched(df_el):
    """
    Checks if a DataFrame element is matched.

    Parameters
    ----------
    df_el : pandas.Series
        A row of the DataFrame.

    Returns
    -------
    pandas.Series or None
        Returns the row if 'matched' is True; otherwise, returns None.
    """
    if df_el['matched'] is True:
        return df_el
    else:
        return None


def not_is_matched(df_el):
    """
    Checks if a DataFrame element is not matched.

    Parameters
    ----------
    df_el : pandas.Series
        A row of the DataFrame.

    Returns
    -------
    pandas.Series or None
        Returns the row if 'matched' is False; otherwise, returns None.
    """
    if df_el['matched'] is False:
        return df_el
    else:
        return None


def split_matched(info_dict: dict | list):
    """
    Separates an info dictionary into matched and not matched items.

    Parameters
    ----------
    info_dict : dict or list
        Info dictionary returned by filtering methods.

    Returns
    -------
    dict
        A dictionary containing DataFrames of matched and not matched items.
    """
    info_out = InfoDict2Pandas(info_dict)
    info_df = info_out['info_df']
    if len(info_df) == 0:
        out = {'info_df': {'matched': None, 'not_matched': None}}
    else:
        matched = info_df.apply(is_matched, axis=1).dropna().copy()
        not_matched = info_df.apply(not_is_matched, axis=1).dropna().copy()
        out = {'info_df': {'matched': matched, 'not_matched': not_matched}}
    if 'matched_ref_df' in info_out.keys():
        ref_df = info_out['matched_ref_df']
        if len(ref_df) == 0:
            out['matched_ref_df'] = {'matched': None, 'not_matched': None}
        else:
            ref_matched = ref_df.apply(is_matched, axis=1).dropna().copy()
            ref_not_matched = ref_df.apply(not_is_matched,
                                           axis=1).dropna().copy()
            out['matched_ref_df'] = {
                'matched': ref_matched, 'not_matched': ref_not_matched}
    return out

# ---------Selenium Utils------------


def initialize_driver(headless: bool = True,
                      incognito: bool = True,
                      add_arguments: list | None = None):
    """
    Initializes a Selenium Chrome Web Driver with specified options.

    Parameters
    ----------
    headless : bool, optional
        If True, runs the driver in headless mode. Defaults to True.
    incognito : bool, optional
        If True, runs the driver in incognito mode. Defaults to True.
    add_arguments : list or None, optional
        Additional arguments to pass to the web driver. Defaults to None.

    Returns
    -------
    webdriver.Chrome
        Initialized Chrome Web Driver instance.

    Raises
    ------
    FailedToAddOptionsArgumentError
        If adding additional arguments to the WebDriver fails.
    DriverInitializationError
        If the driver fails to initialize.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
    if incognito:
        options.add_argument('--incognito')
    if add_arguments:
        for arg in add_arguments:
            try:
                options.add_argument(arg)
            except Exception as e:
                raise errors.FailedToAddOptionsArgumentError(str(e))
    try:
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()
                                                ),
                                options=options)
    except Exception as e:
        raise errors.DriverInitializationError(
            f'Failed to initialize driver with error: {str(e)}')


def page_scroll(driver, n, webpage_wait_time):
    """
    Scrolls a webpage 'n' times using Selenium WebDriver.

    Parameters
    ----------
    driver : webdriver.Chrome
        The Selenium WebDriver instance.
    n : int
        The number of times to scroll the page.
    webpage_wait_time : float
        Time in seconds to wait for the page to load after scrolling.

    Raises
    ------
    ErrorScrollingPage
        If an error occurs while scrolling.
    """
    with logging_redirect_tqdm():
        for i in trange(n, desc=f'Scrolling Page {n} times'):
            try:
                time.sleep(webpage_wait_time)
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            except Exception as e:
                raise errors.ErrorScrollingPage(str(e))
                break


def download_from_src(srcs, prefix, save_path, logger):
    """
    Downloads images from the provided URL sources and checks for corruption.

    Parameters
    ----------
    srcs : list
        List of URL strings to download images from.
    prefix : str
        Prefix string to prepend to each downloaded image filename.
    save_path : str or Path
        Path to the directory where downloaded images will be saved.
    logger : logging.Logger
        Logger instance for logging download activity.

    Raises
    ------
    ImageDownloadError
        If an error occurs during the image downloading process.
    """

    with logging_redirect_tqdm():
        for i, src in tqdm(zip(range(len(srcs)), srcs),
                           desc='Downloading Images...'):
            try:
                s = f'{prefix}_{datetime.now().strftime("%m%d%y%H%M%S%f")}.jpg'
                image_name = save_path / s
                r = requests.get(src)
                with open(image_name, 'wb') as f:
                    f.write(r.content)
                try:
                    with Image.open(image_name) as img:
                        img = img.convert('RGB')
                        image_name.unlink()
                        img.save(image_name, quality=95)
                except Exception as e:
                    logger.warning(f'Image corrupted : {e},deleting')
                    image_name.unlink()
            except requests.exceptions.InvalidSchema:
                logger.error('Unable to open source, Skipping')
                continue
            except Exception as e:
                raise errors.ImageDownloadError(
                    f"Error Downloading Image: {str(e)}")
    logger.info('Images Downloaded Sucessfully')


def img_path_to_tensor(img_path, nsize=None):
    """
    Converts an image at the specified path to a tensor.

    Parameters
    ----------
    img_path : str or Path
        Path to the image file.
    nsize : tuple of int or None, optional
        Size to which to resize the image, if specified. Defaults to None.

    Returns
    -------
    torch.Tensor
        A tensor representation of the image.

    Raises
    ------
    InvalidInputError
        If img_path is not a string or Path object or if the
        file type is not supported.
    """
    torch, v2 = torch_import()
    if isinstance(img_path, str):
        ipath = Path(img_path).resolve()
    elif isinstance(img_path, Path):
        ipath = img_path.resolve()
    else:
        raise errors.InvalidInputError("img_path must be str or Path object")

    if not ipath.suffix.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff',
                                         '.ppm', '.bmp', '.gif')):
        raise errors.InvalidInputError("File Type Not Supported")

    with Image.open(ipath) as img:
        if nsize:
            if not isinstance(nsize, tuple):
                raise errors.InvalidInputError("nsize must be a tuple of ints")
            else:
                transform = v2.Compose([v2.ToImage(), v2.Resize(
                    size=nsize), v2.ToDtype(torch.float32, scale=True)])
                return transform(img)
        else:
            nsize = img.size
            transform = v2.Compose([v2.ToImage(), v2.Resize(
                size=nsize), v2.ToDtype(torch.float32, scale=True)])
            return transform(img)


def dir_path_to_img_batch(path):
    """
     Converts all images in a directory to a batch of tensors.

    Parameters
    ----------
    path : str or Path
        Path to the directory containing image files.

    Returns
    -------
    torch.Tensor
        A tensor representing a batch of images.

    Raises
    ------
    InvalidInputError
        If path is not a valid directory or contains unsupported file types.
    """
    torch, v2 = torch_import()
    if isinstance(path, str):
        dir_path = Path(path).resolve()
    elif isinstance(path, Path):
        dir_path = path.resolve()
    else:
        raise errors.InvalidInputError("path must be str or Path object")

    if not dir_path.is_dir():
        raise errors.InvalidInputError("path is not directory")

    # Convert PNG to RGB
    png_tensors = [img_path_to_tensor(i)[:3, :, :] for i in dir_path.iterdir(
    ) if i.suffix.endswith(".png")]

    tensors = [img_path_to_tensor(i) for i in dir_path.iterdir(
    ) if i.suffix.endswith(".jpg")]

    tensors += png_tensors
    dir_max_width = max([t.shape[1] for t in tensors])
    dir_max_height = max([t.shape[2] for t in tensors])
    transform = v2.Compose([v2.ToImage(), v2.Resize(
        size=(dir_max_width, dir_max_height)),
                            v2.ToDtype(torch.float32, scale=True)])
    r_tensors = [transform(t) for t in tensors]
    batch_list = [x.unsqueeze(0) for x in r_tensors]
    return torch.cat(batch_list)
