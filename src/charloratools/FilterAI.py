#Default Python Libs
import logging
from io import StringIO
import re
import os
import time
import shutil
from pathlib import Path
import copy
from datetime import datetime
import base64
#External Libs
import PIL
from PIL import Image,ImageDraw
import torch
from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import pandas as pd
import imagehash
import hashlib
#Scripts
from . import errors
from . import utils
from . import SysFileManager


class FaceRecognizer:
  """
  Class for using facenet_pytorch pre-trained models to detect and recognize faces.
  Args:
    path:str|Path= Path to a directory containing images
  """

  def __init__(self,path:str|Path):
    """
    :param path str path to directory containing images
    """
    #Setting up Logger
    self.logger=logging.getLogger('FaceRecognizer')

    #Checking torch functionality
    if torch.cuda.is_available():
      self.logger.info(f'Torch is using CUDA\nDevice Name: {torch.cuda.get_device_name()}')
      self.device = torch.device('cuda')
    else:
      self.logger.warning(f'Torch is using CPU!\nAI Model operations will be significantly slower!')
      self.device = torch.device('cpu')

    self.path=Path(path).resolve()
    self.gallery=SysFileManager.GalleryManager(path)

  def __str__(self):
    return f"FaceRecognizer Object,operating on directory:{self.path};\nWith {len(self.gallery)} images;\nUsing torch device:{str(self.device)}"

  def __repr__(self):
    return f"FaceRecognizer(path={self.path})"

  def change_directory(self,path:str|Path):
    """
    Method to change instance's default dir for method usage
    :param path str path to directory containing images
    """
    self.gallery=SysFileManager.GalleryManager(path)

  def filter_images_without_face(self,output_dir:str|Path,min_face_size:int=20,
                                 prob_threshold:float|None=None,return_info=False):
    """
    Method to filter images without face
    :param prob_threshold float threshold for face detection probability,defaults to None
    :param output_dir str path to directory where images will be saved
    :param return_info bool whether to return detections info or not,defaults to False
    :param min_face_size int minimum face size in pixels,defaults to 20
    """
    #Check dir
    output_dir=utils.dirisvalid(path=output_dir,create_if_not_found=True)
    #Initializing model
    mtcnn=MTCNN(keep_all=True,selection_method='probability',min_face_size=min_face_size,device=self.device)
    info=[]
    imgs_without_face=0
    imgs_with_face=0
    #Filtering
    with logging_redirect_tqdm():
      for img_manager in tqdm(self.gallery):
        img=img_manager.path
        img_basename=img_manager.basename
        with Image.open(img) as img_file:
          img_file=img_file.convert('RGB')
          boxes,probs = mtcnn.detect(img_file)
          if boxes is not None:
            if prob_threshold is not None:
              if np.any(probs<prob_threshold):
                self.logger.debug('Face probability is smaller than threshold,skipping')
                imgs_without_face+=1
                info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})
              else:
                self.logger.debug('Face Detected,copying')
                imgs_with_face+=1
                img_file.save(output_dir/img_basename)
                info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': True})
            else:
              self.logger.debug('Face Detected,copying')
              imgs_with_face+=1
              img_file.save(output_dir/img_basename)
              info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': True})
          else:
            self.logger.debug('No face detected,skipping')
            imgs_without_face+=1
            info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})
    if return_info:
      return SysFileManager.GalleryManager(output_dir),{'imgs_with_face':imgs_with_face,'imgs_without_face':imgs_without_face,'info_dict_lst':info}
    else:
      return SysFileManager.GalleryManager(output_dir)

  def filter_images_without_specific_face(self,ref_img_path:str|Path,output_dir:str|Path,prob_threshold:float|None=None,min_face_size:int=20,
                                          distance_threshold:float=0.6,pretrained_model:str='vggface2',distance_function:str='cosine',
                                          return_info:bool=False):
    """
    Method to filter images where the face in the reference image provided is not present
    :param ref_img_path str path to reference image
    :param prob_threshold float threshold for final face detection,defaults to None
    :param output_dir str path to directory where matched images will be saved
    :param distance_function str 'euclidean' or 'cosine' which distance function to use when classifying faces as
           equal or not,defaults to 'cosine'
    :param distance_threshold float distance threshold for considering faces equal, if distance_function='cosine'
           represents the minimum value of cosine similarity the two image tensors must have to be considered images
           containing the same face. if distance_function='euclidean' represents the maximum value of euclidean distance
           the two image tensors are allowed to have in order to be considered images of the same face,defaults to 0.6
           since distance_fucntion defaults to cosine
    :param return_info bool whether to return detections info or not,defaults to False
    :param min_face_size int minimum face size in pixels,defaults to 20
    :param pretrained_model str which pretrained model from facenet_pytorch for Resnet,defaults to vggface2
    """
    #Check pretained_arg
    pretrained_available=['vggface2','casia-webface']
    if pretrained_model not in pretrained_available:
      raise errors.InvalidInputError("pretrained model must be one of ['vggface2','casia-webface']")
    #Check distance_function
    distance_available=['euclidean','cosine']
    if distance_function not in distance_available:
      raise errors.InvalidInputError("distance_function must be one of ['euclidean','cosine']")
    #Check ref_img
    ref_img_manager=SysFileManager.ImgManager(ref_img_path)
    #Check dir
    output_dir=utils.dirisvalid(path=output_dir,create_if_not_found=True)
    imgs_with_ref_face=0
    imgs_without_ref_face=0
    info=[]
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True,selection_method='probability',min_face_size=min_face_size,device=self.device)
    # Load pre-trained FaceNet model
    resnet = InceptionResnetV1(pretrained=pretrained_model).eval()
    ref_img=cv2.imread(ref_img_manager.path)
    ref_img=cv2.cvtColor(ref_img,cv2.COLOR_BGR2RGB)
    ref_boxes,probs= mtcnn.detect(ref_img)
    if ref_boxes is None:
      raise errors.NoFaceDetectedInReferenceImage("No Face was detected in Reference Image")
    # Get embedding for the reference face
    ref_aligned=mtcnn(ref_img)
    ref_embeddings=resnet(ref_aligned)
    with logging_redirect_tqdm():
      for img_manager in tqdm(self.gallery):
        img_path=img_manager.path
        img_basename=img_manager.basename
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Get embeddings for the current image
        boxes,probs=mtcnn.detect(img)
        if boxes is None:
          self.logger.debug('No Face in Image, skipping')
          info.append({'path': img_path,'boxes':boxes,'probs': probs, 'matched': False})
          imgs_without_ref_face+=1
        else:
          # Detect faces and extract embeddings
          if prob_threshold is not None:
            if np.any(probs<prob_threshold):
              self.logger.debug('Face probability is smaller than threshold,skipping')
              imgs_without_ref_face+=1
              info.append({'path': img_path, 'boxes': boxes, 'probs': probs, 'matched': False})
            else:
              faces = mtcnn.detect(img)
              aligned = mtcnn(img)
              embeddings = resnet(aligned)
              # Calculate the distance between embeddings and classify as same or not
              distance=utils.distance_function(ref_embeddings,embeddings,method=distance_function)
              same_face=utils.distance_function(ref_embeddings,embeddings,method=distance_function,classify=True,threshold=distance_threshold)
              if not same_face:
                info.append({'path': img_path,'boxes':boxes,'probs': probs, 'distance': distance, 'matched': False})
                imgs_without_ref_face += 1
                self.logger.debug("Face doesn't match,skipping")
              else:
                info.append({'path': img_path,'boxes':boxes,'probs': probs, 'distance': distance, 'matched': True})
                self.logger.debug("Face matches,copying")
                imgs_with_ref_face += 1
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_dir/img_basename, rgb_img)

          else:
            faces=mtcnn.detect(img)
            aligned=mtcnn(img)
            embeddings=resnet(aligned)
            # Calculate the distance between embeddings and classify as same or not
            distance = utils.distance_function(ref_embeddings, embeddings, method=distance_function)
            same_face = utils.distance_function(ref_embeddings, embeddings, method=distance_function,classify=True, threshold=distance_threshold)
            if not same_face:
              info.append({'path': img_path,'boxes':boxes,'probs': probs, 'distance': distance, 'matched': False})
              imgs_without_ref_face+=1
              self.logger.debug("Face doesn't match,skipping")
            else:
              info.append({'path': img_path,'boxes':boxes,'probs': probs, 'distance': distance, 'matched': True})
              self.logger.debug("Face matches,copying")
              imgs_with_ref_face+=1
              rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
              cv2.imwrite(output_dir/img_basename,rgb_img)

    if return_info:
      return SysFileManager.GalleryManager(output_dir),{'imgs_with_ref_face':imgs_with_ref_face,'imgs_without_ref_face':imgs_without_ref_face,'info_dict_lst':info}
    else:
      return SysFileManager.GalleryManager(output_dir)

  def filter_images_with_multiple_faces(self,output_dir:str|Path,prob_threshold:float|None=None,min_face_size:int=20,return_info:bool=False):
    """
    Method to filter images where the number of detected faces is greater than one
    :param prob_threshold float|None final probability threshold for face detection,defaults to None
    :param min_face_size int|None minimum size of face detection,defaults to 20
    :param output_dir str directory to save filtered images
    :param return_info bool whether to return filtered images info,defaults to False
    """
    #Check dir
    output_dir=utils.dirisvalid(path=output_dir,create_if_not_found=True)
    mtcnn=MTCNN(keep_all=True,selection_method='probability',min_face_size=min_face_size,device=self.device)
    info=[]
    imgs_without_face=0
    imgs_with_multiple_faces=0
    imgs_with_one_face=0
    with logging_redirect_tqdm():
      for img_manager in tqdm(self.gallery):
        img=img_manager.path
        img_basename=img_manager.basename
        with Image.open(img) as img_file:
          img_file=img_file.convert('RGB')
          boxes,probs = mtcnn.detect(img_file)
          if boxes is not None:
            if prob_threshold is not None:
              if np.any(probs<prob_threshold):
                self.logger.debug('Face probability is smaller than threshold,skipping')
                imgs_without_face+=1
                info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})
              else:
                if len(boxes)>1:
                  self.logger.debug('Multiple faces detected,skipping')
                  imgs_with_multiple_faces+=1
                  info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})
                else:
                  self.logger.debug('Face Detected,copying')
                  imgs_with_one_face+=1
                  img_file.save(output_dir/img_basename)
                  info.append({'path': img,'boxes':boxes,'probs': probs,'matched': True})
            else:
              if len(boxes) > 1:
                self.logger.debug('Multiple faces detected,skipping')
                imgs_with_multiple_faces += 1
                info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})
              else:
                self.logger.debug('Face Detected,copying')
                imgs_with_one_face += 1
                img_file.save(output_dir/img_basename)
                info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': True})
          else:
            self.logger.debug('No face detected,skipping')
            imgs_without_face+=1
            info.append({'path': img, 'boxes': boxes, 'probs': probs, 'matched': False})

      if return_info:
        return SysFileManager.GalleryManager(output_dir),{'imgs_without_face':imgs_without_face,'imgs_with_multiple_faces':imgs_with_multiple_faces,'imgs_with_one_face':imgs_with_one_face,'info_dict_lst':info}
      else:
        return SysFileManager.GalleryManager(output_dir)

  def save_images_with_detection_box(self,info_dict_lst:list,output_dir:str|Path,save_only_matched:bool=False):
    """
    Method to save images with model's detection boxes drawn with red outline to an output directory
    from an 'info' dict list (dict with same format as the one returned by one of the filtering methods)
    format:

    info_dict_lst=[{
    'path': File path of image
    'boxes': Numpy array of detection boxes returned by the model or None (if no face was detected)
    'matched': Boolean of wether the image passed the filter of whichever filtering method generated the dict
    '...':'...'},
    {...},...]

    -All key-value pairs specified above must exist on every dict of info_dict_lst.
    -Any other key-value pair other than the ones specified above will be ignored and do not need to be present.

    The images in which the model didn't recognize a face ('boxes':None) will be skipped

    :param info_dict_lst: list specified above in docstring
    :param output_dir: str Directory to save images to
    :param save_only_matched:bool Wether to save only matched images ('matched':True),defaults to False
    """

    for d in info_dict_lst:
      if 'distance' in d.keys():
        if isinstance(d['distance'],torch.Tensor):
          d['distance'] = d['distance'].detach().cpu().numpy()
      if isinstance(d['boxes'],list):
        d['boxes']=np.array(d['boxes'])
      if isinstance(['boxes'],np.ndarray):
        if np.all(d['boxes']==None):
          d['boxes']=None

    info_dict_lst=copy.deepcopy(info_dict_lst)
    if save_only_matched:
      selected_dicts=[i for i in info_dict_lst if i['boxes'] is not None and i['matched'] is True]
    else:
      selected_dicts=[i for i in info_dict_lst if i['boxes'] is not None]

    for sdict in tqdm(selected_dicts):
      utils.save_with_detection_box(sdict['path'],output_dir,sdict['boxes'])
    self.logger.info('Images with detection boxes drawn saved sucessfully')
    return SysFileManager.GalleryManager(output_dir)
