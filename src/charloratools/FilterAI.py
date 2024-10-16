"""
Functionalities
---------------
  - face detection and recognition using the `facenet_pytorch`
  - Processing images in bulk
  - Filtering images based on face detection
  - Saving images with detection overlays.

Classes
-------
FaceRecognizer
    A class that utilizes pre-trained models from `facenet_pytorch`
    to detect and recognize faces in a set of images.


Raises
------
TorchNotInstalledError
    Raised when the required PyTorch library is not found.
InvalidInputError
    Raised for invalid inputs related to image paths and parameters.
NoFaceDetectedInReferenceImage
    Raised when no face is detected in the provided reference image.

Examples
--------
```python
from charloratools.FilterAI import FaceRecognizer
fr = FaceRecognizer('path/to/images')
fr.filter_images_without_face('output/directory')
```
"""

# Default Python Libs
import logging
from pathlib import Path
import copy
# External Libs
from PIL import Image
from .facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
# Scripts
from . import errors
from . import utils
from . import SysFileManager


class FaceRecognizer:
    """
    Class for using facenet_pytorch pre-trained models to detect
    and recognize faces.
    """

    def __init__(self, path: str | Path):
        """
        Initializes the FaceRecognizer with the specified image path.

        Parameters
        ----------
        path : str or Path
            Path to the directory containing images.

        Raises
        ------
        TorchNotInstalledError
            If PyTorch is not installed on the system.
        """
        try:
            import torch
        except ImportError:
            raise errors.TorchNotInstalledError("Torch is not installed.")
        # Setting up Logger
        self.logger = logging.getLogger('FaceRecognizer')

        # Checking torch functionality
        if torch.cuda.is_available():
            self.logger.info(
                f'''Torch is using CUDA
                    Device Name: {torch.cuda.get_device_name()}''')
            self.device = torch.device('cuda')
        else:
            self.logger.warning(
                '''Torch is using CPU!\nAI Model operations will be
                    significantly slower!''')
            self.device = torch.device('cpu')

        self.path = Path(path).resolve()
        self.gallery = SysFileManager.GalleryManager(path)

    def __str__(self):
        """
        Returns a string representation of the FaceRecognizer object.

        Returns
        -------
        str
            A descriptive string of the FaceRecognizer, including the path,
            number of images, and used torch device.
        """
        s = f"""
        FaceRecognizer Object,operating on directory:{self.path};
        With {len(self.gallery)} images;
        Using torch device:{str(self.device)}
        """
        return s

    def __repr__(self):
        """
        Returns a detailed string representation of the
        FaceRecognizer instance.

        Returns
        -------
        str
            A representation string for the FaceRecognizer instance.
        """
        return f"FaceRecognizer(path={self.path})"

    def change_directory(self, path: str | Path):
        """
        Changes the directory of the images used by the FaceRecognizer.

        Parameters
        ----------
        path : str or Path
            New path to the directory containing images.
        """
        self.gallery = SysFileManager.GalleryManager(path)

    def filter_images_without_face(self, output_dir: str | Path,
                                   min_face_size: int = 20,
                                   prob_threshold: float | None = None,
                                   return_info=False):
        """
        Filters images that do not contain detected faces.

        Parameters
        ----------
        output_dir : str or Path
            Path to the directory where images will be saved.
        min_face_size : int, optional
            Minimum face size in pixels for detection. Defaults to 20.
        prob_threshold : float, optional
            Threshold for face detection probability. Defaults to None.
        return_info : bool, optional
            If True, returns detection information. Defaults to False.

        Returns
        -------
        SysFileManager.GalleryManager or tuple
            Returns the GalleryManager for the output directory.
            If `return_info` is True,
            returns a tuple containing the GalleryManager object
            and a dictionary with filtering info.

        Raises
        ------
        InvalidPathError
            If the specified output directory is invalid.
        """
        # Check dir
        output_dir = utils.dirisvalid(
            path=output_dir, create_if_not_found=True)
        # Initializing model
        mtcnn = MTCNN(keep_all=True, selection_method='probability',
                      min_face_size=min_face_size, device=self.device)
        info = []
        imgs_without_face = 0
        imgs_with_face = 0
        # Filtering
        with logging_redirect_tqdm():
            for img_manager in tqdm(self.gallery):
                img = img_manager.path
                img_basename = img_manager.basename
                with Image.open(img) as img_file:
                    img_file = img_file.convert('RGB')
                    boxes, probs = mtcnn.detect(img_file)
                    if boxes is not None:
                        if prob_threshold is not None:
                            if np.any(probs < prob_threshold):
                                self.logger.debug(
                                    'probability is smaller than threshold')
                                imgs_without_face += 1
                                info.append(
                                    {'path': img,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'matched': False})
                            else:
                                self.logger.debug('Face Detected,copying')
                                imgs_with_face += 1
                                img_file.save(output_dir/img_basename)
                                info.append(
                                    {'path': img,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'matched': True})
                        else:
                            self.logger.debug('Face Detected,copying')
                            imgs_with_face += 1
                            img_file.save(output_dir/img_basename)
                            info.append(
                                {'path': img,
                                 'boxes': boxes,
                                 'probs': probs,
                                 'matched': True})
                    else:
                        self.logger.debug('No face detected,skipping')
                        imgs_without_face += 1
                        info.append({'path': img, 'boxes': boxes,
                                    'probs': probs, 'matched': False})
        if return_info:
            return (SysFileManager.GalleryManager(output_dir),
                    {'imgs_with_face': imgs_with_face,
                     'imgs_without_face': imgs_without_face,
                     'info_dict_lst': info})
        else:
            return SysFileManager.GalleryManager(output_dir)

    def filter_images_without_specific_face(self,
                                            ref_img_path: str | Path,
                                            output_dir: str | Path,
                                            prob_threshold: float = None,
                                            min_face_size: int = 20,
                                            distance_threshold: float = 0.6,
                                            pretrained_model: str = 'vggface2',
                                            distance_function: str = 'cosine',
                                            return_info: bool = False):
        """
        Filters images that do not contain a specific face.

        Parameters
        ----------
        ref_img_path : str or Path
            Path to the reference image containing the face to match.
        output_dir : str or Path
            Path to the directory where matched images will be saved.
        prob_threshold : float, optional
            Threshold for final face detection probability. Defaults to None.
        min_face_size : int, optional
            Minimum face size in pixels for detection. Defaults to 20.
        distance_threshold : float, optional
            Threshold for considering faces equal. Defaults to 0.6.
        pretrained_model : str, optional
            Pre-trained model to use from facenet_pytorch.
            Defaults to 'vggface2'.
        distance_function : str, optional
            Distance function for matching 'euclidean' or 'cosine'.
            Defaults to 'cosine'.
        return_info : bool, optional
            If True, returns detection information. Defaults to False.

        Returns
        -------
        SysFileManager.GalleryManager or tuple
            Returns the GalleryManager for the output directory.
            If `return_info` is True,
            returns a tuple containing the GalleryManager object
            and a dictionary with filtering info.

        Raises
        ------
        InvalidInputError
            If the provided model or distance function is invalid.
        NoFaceDetectedInReferenceImage
            If no face is detected in the reference image.
        InvalidPathError
            If the specified output directory is invalid.
        """
        # Check pretained_arg
        pretrained_available = ['vggface2', 'casia-webface']
        if pretrained_model not in pretrained_available:
            raise errors.InvalidInputError(
                "pretrained model must be one of ['vggface2','casia-webface']")
        # Check distance_function
        distance_available = ['euclidean', 'cosine']
        if distance_function not in distance_available:
            raise errors.InvalidInputError(
                "distance_function must be one of ['euclidean','cosine']")
        # Check ref_img
        ref_img_manager = SysFileManager.ImgManager(ref_img_path)
        # Check dir
        output_dir = utils.dirisvalid(
            path=output_dir, create_if_not_found=True)
        imgs_with_ref_face = 0
        imgs_without_ref_face = 0
        info = []
        # Initialize MTCNN for face detection
        mtcnn = MTCNN(keep_all=True, selection_method='probability',
                      min_face_size=min_face_size, device=self.device)
        # Load pre-trained FaceNet model
        resnet = InceptionResnetV1(pretrained=pretrained_model).eval()
        ref_img = cv2.imread(ref_img_manager.path)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_boxes, probs = mtcnn.detect(ref_img)
        if ref_boxes is None:
            raise errors.NoFaceDetectedInReferenceImage(
                "No Face was detected in Reference Image")
        # Get embedding for the reference face
        ref_aligned = mtcnn(ref_img)
        ref_embeddings = resnet(ref_aligned)
        with logging_redirect_tqdm():
            for img_manager in tqdm(self.gallery):
                img_path = img_manager.path
                img_basename = img_manager.basename
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Get embeddings for the current image
                boxes, probs = mtcnn.detect(img)
                if boxes is None:
                    self.logger.debug('No Face in Image, skipping')
                    info.append({'path': img_path, 'boxes': boxes,
                                'probs': probs, 'matched': False})
                    imgs_without_ref_face += 1
                else:
                    # Detect faces and extract embeddings
                    if prob_threshold is not None:
                        if np.any(probs < prob_threshold):
                            self.logger.debug(
                                'probability is smaller than threshold')
                            imgs_without_ref_face += 1
                            info.append(
                                {'path': img_path,
                                 'boxes': boxes,
                                 'probs': probs,
                                 'matched': False})
                        else:
                            aligned = mtcnn(img)
                            embeddings = resnet(aligned)
                            # Calculate the distance between embeddings and
                            # classify as same or not
                            distance = utils.distance_function(
                                ref_embeddings, embeddings,
                                method=distance_function)
                            same_face = utils.distance_function(
                                ref_embeddings, embeddings,
                                method=distance_function,
                                classify=True,
                                threshold=distance_threshold)
                            if not same_face:
                                info.append(
                                    {'path': img_path,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'distance': distance,
                                     'matched': False})
                                imgs_without_ref_face += 1
                                self.logger.debug(
                                    "Face doesn't match,skipping")
                            else:
                                info.append(
                                    {'path': img_path,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'distance': distance,
                                     'matched': True})
                                self.logger.debug("Face matches,copying")
                                imgs_with_ref_face += 1
                                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                cv2.imwrite(output_dir/img_basename, rgb_img)

                    else:
                        aligned = mtcnn(img)
                        embeddings = resnet(aligned)
                        # Calculate the distance between embeddings and
                        # classify as same or not
                        distance = utils.distance_function(
                            ref_embeddings, embeddings,
                            method=distance_function)
                        same_face = utils.distance_function(
                            ref_embeddings, embeddings,
                            method=distance_function,
                            classify=True,
                            threshold=distance_threshold)
                        if not same_face:
                            info.append(
                                {'path': img_path,
                                 'boxes': boxes,
                                 'probs': probs,
                                 'distance': distance,
                                 'matched': False})
                            imgs_without_ref_face += 1
                            self.logger.debug("Face doesn't match,skipping")
                        else:
                            info.append(
                                {'path': img_path,
                                 'boxes': boxes,
                                 'probs': probs,
                                 'distance': distance,
                                 'matched': True})
                            self.logger.debug("Face matches,copying")
                            imgs_with_ref_face += 1
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(output_dir/img_basename, rgb_img)

        if return_info:
            return (SysFileManager.GalleryManager(output_dir),
                    {'imgs_with_ref_face': imgs_with_ref_face,
                     'imgs_without_ref_face': imgs_without_ref_face,
                     'info_dict_lst': info})
        else:
            return SysFileManager.GalleryManager(output_dir)

    def filter_images_with_multiple_faces(self,
                                          output_dir: str | Path,
                                          prob_threshold: float | None = None,
                                          min_face_size: int = 20,
                                          return_info: bool = False):
        """
        Filters images with more than one detected face.

        Parameters
        ----------
        output_dir : str or Path
            Path to the directory where images will be saved.
        prob_threshold : float, optional
            Probability threshold for face detection. Defaults to None.
        min_face_size : int, optional
            Minimum face size in pixels for detection. Defaults to 20.
        return_info : bool, optional
            If True, returns detection information. Defaults to False.

        Returns
        -------
        SysFileManager.GalleryManager or tuple
            Returns the GalleryManager for the output directory.
            If `return_info` is True,
            returns a tuple containing the GalleryManager object
            and a dictionary with filtering info.

        Raises
        ------
        InvalidPathError
            If the specified output directory is invalid.
        """
        # Check dir
        output_dir = utils.dirisvalid(
            path=output_dir, create_if_not_found=True)
        mtcnn = MTCNN(keep_all=True, selection_method='probability',
                      min_face_size=min_face_size, device=self.device)
        info = []
        imgs_without_face = 0
        imgs_with_multiple_faces = 0
        imgs_with_one_face = 0
        with logging_redirect_tqdm():
            for img_manager in tqdm(self.gallery):
                img = img_manager.path
                img_basename = img_manager.basename
                with Image.open(img) as img_file:
                    img_file = img_file.convert('RGB')
                    boxes, probs = mtcnn.detect(img_file)
                    if boxes is not None:
                        if prob_threshold is not None:
                            if np.any(probs < prob_threshold):
                                self.logger.debug(
                                    'probability is smaller than threshold')
                                imgs_without_face += 1
                                info.append(
                                    {'path': img,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'matched': False})
                            else:
                                if len(boxes) > 1:
                                    self.logger.debug(
                                        'Multiple faces detected,skipping')
                                    imgs_with_multiple_faces += 1
                                    info.append(
                                        {'path': img,
                                         'boxes': boxes,
                                         'probs': probs,
                                         'matched': False})
                                else:
                                    self.logger.debug('Face Detected,copying')
                                    imgs_with_one_face += 1
                                    img_file.save(output_dir/img_basename)
                                    info.append(
                                        {'path': img,
                                         'boxes': boxes,
                                         'probs': probs,
                                         'matched': True})
                        else:
                            if len(boxes) > 1:
                                self.logger.debug(
                                    'Multiple faces detected,skipping')
                                imgs_with_multiple_faces += 1
                                info.append(
                                    {'path': img,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'matched': False})
                            else:
                                self.logger.debug('Face Detected,copying')
                                imgs_with_one_face += 1
                                img_file.save(output_dir/img_basename)
                                info.append(
                                    {'path': img,
                                     'boxes': boxes,
                                     'probs': probs,
                                     'matched': True})
                    else:
                        self.logger.debug('No face detected,skipping')
                        imgs_without_face += 1
                        info.append({'path': img, 'boxes': boxes,
                                    'probs': probs, 'matched': False})

            if return_info:
                return (SysFileManager.GalleryManager(output_dir),
                        {'imgs_without_face': imgs_without_face,
                         'imgs_with_multiple_faces': imgs_with_multiple_faces,
                         'imgs_with_one_face': imgs_with_one_face,
                         'info_dict_lst': info})
            else:
                return SysFileManager.GalleryManager(output_dir)

    def save_images_with_detection_box(self,
                                       info_dict_lst: list,
                                       output_dir: str | Path,
                                       save_only_matched: bool = False):
        """
        Saves images with detection boxes drawn to an output directory.

        This method takes an info dictionary list
        (from previous filtering methods) and saves the images
        with the model's detection boxes as red outlines.

        Parameters
        ----------
        info_dict_lst : list
            A list of dictionaries with the following keys:
            - 'path': File path of the image.
            - 'boxes': Numpy array of detection boxes returned by the model
                       or None.
            - 'matched': Boolean indicating if the image passed the filter.
        output_dir : str or Path
            Directory to save images with boxes drawn.
        save_only_matched : bool, optional
            If True, saves only matched images
            (i.e., those where 'matched' is True). Defaults to False.

        Returns
        -------
        SysFileManager.GalleryManager
            Manager for the directory containing saved images.

        Raises
        ------
        TorchNotInstalledError
            If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError:
            raise errors.TorchNotInstalledError("Torch is not installed.")
        for d in info_dict_lst:
            if 'distance' in d.keys():
                if isinstance(d['distance'], torch.Tensor):
                    d['distance'] = d['distance'].detach().cpu().numpy()
            if isinstance(d['boxes'], list):
                d['boxes'] = np.array(d['boxes'])
            if isinstance(['boxes'], np.ndarray):
                if np.all(d['boxes'] is None):
                    d['boxes'] = None

        info_dict_lst = copy.deepcopy(info_dict_lst)
        if save_only_matched:
            selected_dicts = []
            for i in info_dict_lst:
                if i['boxes'] is not None and i['matched'] is True:
                    selected_dicts.append(i)
        else:
            selected_dicts = [
                i for i in info_dict_lst if i['boxes'] is not None]

        for sdict in tqdm(selected_dicts):
            utils.save_with_detection_box(
                sdict['path'], output_dir, sdict['boxes'])
        self.logger.info('Images with detection boxes drawn saved sucessfully')
        return SysFileManager.GalleryManager(output_dir)
