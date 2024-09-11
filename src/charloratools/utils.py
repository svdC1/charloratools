import logging
from io import StringIO
import re
import os
import time
import requests
import shutil
from pathlib import Path
import copy
from datetime import datetime
import base64
#External Libs
import PIL
from PIL import Image,ImageDraw
import torch
from torchvision.transforms import v2
from facenet_pytorch import MTCNN,InceptionResnetV1
import cv2
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import pandas as pd
import imagehash
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm,trange
#Scripts
from . import errors
from . import SysFileManager

def dirisvalid(path:str,check_images:bool=True,return_info:bool=False,hashtype:str|None=None,create_if_not_found:bool=False,show_tqdm:bool=False):
  """
    Function to check dir properties or create new directory if it doens't exist
    :param path str path to directory
    :param check_images bool whether to check if at least 1 image exists or not,defaults to True
    :param return_info bool whether to return number of images in dir and their paths or not,defaults to False
    :param hashtype str hash type to use for hashing imgs,must be provided if return_info=True,defaults to None
    :param create_if_not_found bool whether to create directory if it does not exist,defaults to False
    :param show_tqdm bool wether or not to show image loading and hashing progress with tqdm,defaults to False
  """
  logger=logging.getLogger(__name__)
  if return_info and hashtype is None:
    raise errors.InvalidInputError("hashtype must be provided if return_info=True")

  if isinstance(path,str):
    path = Path(path).resolve()
  elif isinstance(path,Path):
    path= path
  else:
    raise errors.InvalidTypeError("Path must be a string or path-like object")
  #Create Directory and send path if it isn't found and arg is set to true
  if not path.exists():
    if create_if_not_found:
      path.mkdir(parents=True,exist_ok=True)
      path=path.resolve()
      logger.debug(f'Created empty directory {path.resolve()}')
      return path
    else:
      raise errors.InvalidPathError("Path does not exist")
  #If arg is set to true and dir exists return path
  else:
    if create_if_not_found:
      return path
  #If not img arg is set to False validate
  if not check_images:
    return True
  #Checking Images
  img_paths=[path/i for i in path.iterdir() if i.suffix.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.ppm', '.bmp', '.gif'))]
  if len(img_paths)==0:
    raise errors.NoImagesInDirectoryError("No images found in directory")
  #Checking if pillow can open the image,else deleting
  images={}
  if show_tqdm:
    with logging_redirect_tqdm():
      for img_path in tqdm(img_paths,desc='Checking and Hashing Images'):
        try:
          with Image.open(img_path) as img:
            #Checking if operation can be performed on img
            np.array(img)
        except PIL.UnidentifiedImageError:
          img_path.unlink()
          logger.debug(f'Removed {img_path} due to PIL error')

        images[img_path]=SysFileManager.ImgManager(img_path,hashtype=hashtype)
  else:
    for img_path in img_paths:
      try:
        with Image.open(img_path) as img:
          #Checking if operation can be performed on img
          np.array(img)
      except PIL.UnidentifiedImageError:
        img_path.unlink()
        logger.debug(f'Removed {img_path} due to PIL error')

      images[img_path]=SysFileManager.ImgManager(img_path,hashtype=hashtype)

  if return_info:
    logger.debug("Directory is Valid")
    logger.debug(f"Found {len(images)} images in directory")
    return path,images
  else:
    logger.debug("Directory is Valid")
    return True

def GetUniqueDtStr():
  return datetime.now().strftime("%m%d%y%H%M%S%f")

#---------Face Recognizer Utils------------
def save_with_detection_box(img_path:str,outdir:str,boxes):
  outdir=dirisvalid(outdir,create_if_not_found=True)
  with Image.open(img_path) as img:
    draw=ImageDraw.Draw(img)
    for box in boxes:
      draw.rectangle(box.tolist(),outline='red',width=5)
    if img_path.endswith('.jpg'):
      img.save(outdir/img_path.stem,quality=95)
    else:
      img.save(outdir/img_path.stem)

def distance_function(embedding1, embedding2, method, classify=False, threshold=None):
  """
  Function for calculating the distance between two embeddings using either Euclidean Distance or
  PyTorch builtin Functional Cosine Similarity Function
  Euclidean Distance: Ranges dependant of tensor values
  Cosine Similiarity: 1=identical vectors;0=orthogonal vectors (no similarity),-1=completely opposite vectors.
  :param embedding1: torch tensor 1
  :param embedding2: torch tensor 2
  :param method: str 'euclidean' or 'cosine' distance function to use
  :param classify: wether to classify embeddings as matching or not based on provided threshold,defaults to False
  :param threshold: threshold for minimum cosine similarity or maximum euclidean distance classsification,defaults to None
  """
  supported_methods = ['euclidean', 'cosine']
  if method not in supported_methods:
    raise errors.InvalidInputError('Method not supported')
  if classify and threshold is None:
    raise errors.InvalidInputError('Threshold must be provided when classify is set to True')
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
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    if classify:
      if torch.any(cosine_similarity >= threshold):
        # Matches
        return True
      else:
        return False
    else:
      return cosine_similarity

def InfoDict2Pandas(info:dict|list):
  """
  Function to validate and convert info dict returned by FaceRecognizer filtering methods to formatted Pandas DataFrame
  :param info: dict returned by FaceRecognizer filtering methods or list of dicts present in 'info_dict'
   """
  if not isinstance(info, list) and not isinstance(info,dict):
    raise errors.InfoDictFormatError('Must be a dict or list of dicts')
  possible_outer_keys=['imgs_with_face','imgs_without_face','imgs_with_ref_face','imgs_without_ref_face','imgs_with_multiple_faces','imgs_with_one_face']
  expected_inner_keys=['path','boxes','probs','matched']
  #Validating Input
  if type(info)==dict:
    if 'info_dict_lst' in info.keys():
      if not isinstance(info['info_dict_lst'],list):
        raise errors.InfoDictFormatError('info_dict_lst Must be a list of dicts')
      for d in info['info_dict_lst']:
        if not isinstance(d,dict):
          raise errors.InfoDictFormatError('info_dict_lst must be a list of dicts')
        for key in expected_inner_keys:
          if key not in d.keys():
            raise errors.InfoDictFormatError('One of the expected keys of info_dict_lst was not found in one of the dicts')
        if 'distance' in d.keys():
          if isinstance(d['distance'],torch.Tensor):
            d['distance'] = d['distance'].detach().cpu().numpy()
    else:
      raise errors.InfoDictFormatError('Did not find info_dict_lst key')
  else:
    for d in info['info_dict_lst']:
      if not isinstance(d, dict):
        raise errors.InfoDictFormatError('info_dict_lst must be a list of dicts')
      for key in expected_inner_keys:
        if key not in d.keys():
          raise errors.InfoDictFormatError('One of the expected keys of info_dict_lst was not found in one of the dicts')
      if 'distance' in d.keys():
        if isinstance(d['distance'], torch.Tensor):
          d['distance'] = d['distance'].detach().cpu().numpy()
  #----
  #---- Normalizing Boxes None format
  if type(info)==dict:
    for d in info['info_dict_lst']:
      if isinstance(d['boxes'],list):
        d['boxes']=np.array(d['boxes'])
      if isinstance(d['boxes'],np.ndarray):
        if np.any(d['boxes']==None):
          d['boxes']=None
  else:
    for d in info:
      if isinstance(d['boxes'],list):
        d['boxes']=np.array(d['boxes'])
      if isinstance(d['boxes'],np.ndarray):
        if np.any(d['boxes']==None):
          d['boxes']=None
  #---- Sorting info
  fdict={}
  fdistance_dict={}
  pkeys_present={}
  if type(info)==dict:
    for d in info['info_dict_lst']:
      if 'distance' in d.keys():
        for k, v in d.items():
          if k not in fdistance_dict.keys():
            fdistance_dict[k] = [v]
          else:
            fdistance_dict[k].append(v)
      else:
        for k,v in d.items():
          if k not in fdict.keys():
            fdict[k] = [v]
          else:
            fdict[k].append(v)

    for k,v in info.items():
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
  #---- Converting to Pandas
  out={}
  if len(fdistance_dict)>0:
    out['matched_ref_df']=pd.DataFrame(fdistance_dict)
  if len(pkeys_present)>0:
    out['n_filtered_series']=pd.Series(pkeys_present)
  out['info_df']=pd.DataFrame(fdict)
  return out

def split_matched(info_dict:dict|list):
  """
  Separates an info dict returned by FaceRecognizer filtering methods to Pandas DataFrames of matched and not matched
  items
  :param info_dict: dict or lst of dicts returned by FaceRecognizer filtering methods
  """
  info_out=InfoDict2Pandas(info_dict)
  info_df = info_out['info_df']
  if len(info_df)==0:
    out={'info_df':{'matched':None,'not_matched':None}}
  else:
    matched=info_df[info_df['matched']==True].copy()
    not_matched=info_df[info_df['matched']==False].copy()
    out={'info_df':{'matched':matched,'not_matched':not_matched}}
  if 'matched_ref_df' in info_out.keys():
    ref_df=info_out['matched_ref_df']
    if len(ref_df)==0:
      out['matched_ref_df']={'matched':None,'not_matched':None}
    else:
      ref_matched = ref_df[ref_df['matched'] == True].copy()
      ref_not_matched = ref_df[ref_df['matched'] == False].copy()
      out['matched_ref_df']={'matched':ref_matched,'not_matched':ref_not_matched}
  return out

#---------Selenium Utils------------
def initialize_driver(headless:bool=True,incognito:bool=True,add_arguments:list|None=None):
  """
  Funtion to initialize a Selenium Chrome Web Driver
  :param headless: Wether to initialize the webdriver in headless mode or not
  :param incognito: Wether to initialize in incognito mode or not
  :param add_arguments: Additional arguments to pass to the webdriver
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
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
  except Exception as e:
    raise errors.DriverInitializationError(f'Failed to initialize driver with error: {str(e)}')

def page_scroll(driver,n,webpage_wait_time):
  """
    Function to scroll a page 'n' times using Selenium WebDriver
    :param driver: Selenium WebDriver
    :param n: Number of times to scroll the page
    :param webpage_wait_time: Time to wait for the page to load
  """
  with logging_redirect_tqdm():
    for i in trange(n,desc=f'Scrolling Page {n} times'):
      try:
        time.sleep(webpage_wait_time)
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
      except Exception as e:
        raise errors.ErrorScrollingPage(str(e))
        break

def download_from_src(srcs,prefix,save_path,logger):
  """
  Function to download images from url sources and check if they're corrupted using requests and Pillow
  :param srcs: Url sources to download
  :param prefix: Prefix string to prepend to each img downloaded
  :param save_path: Path to save the downloaded images
  :param logger: logging.Logger instance
  :param insta_pics: Wether to apply routine for instagram pics
  """

  with logging_redirect_tqdm():
    for i, src in tqdm(zip(range(len(srcs)), srcs),desc='Downloading Images...'):
      try:
        image_name = save_path/f'{prefix}_{datetime.now().strftime("%m%d%y%H%M%S%f")}.jpg'
        r = requests.get(src)
        with open(image_name, 'wb') as f:
          f.write(r.content)
        try:
          with Image.open(image_name) as img:
            img=img.convert('RGB')
            image_name.unlink()
            img.save(image_name,quality=95)
        except:
          logger.warning(f'Image corrupted,deleting')
          image_name.unlink()
      except requests.exceptions.InvalidSchema:
        logger.error(f'Unable to open source, Skipping')
        continue
      except Exception as e:
        raise errors.ImageDownloadError(f"Error Downloading Image: {str(e)}")
  logger.info('Images Downloaded Sucessfully')


def img_path_to_tensor(img_path,nsize=None):
  if isinstance(img_path,str):
    ipath=Path(img_path).resolve()
  elif isinstance(img_path,Path):
    ipath=img_path.resolve()
  else:
    raise errors.InvalidInputError("img_path must be str or Path object")

  with Image.open(ipath) as img:
    if nsize:
      if not isinstance(nsize,tuple):
        raise errors.InvalidInputError("nsize must be a tuple of ints")
      else:
        transform=v2.Compose([v2.ToImage(),v2.Resize(size=nsize),v2.ToDtype(torch.float32,scale=True)])
        return transform(img)
    else:
      nsize=img.size
      transform=v2.Compose([v2.ToImage(),v2.Resize(size=nsize),v2.ToDtype(torch.float32,scale=True)])
      return transform(img)


def dir_path_to_img_batch(path):
  if isinstance(path,str):
    dir_path=Path(path).resolve()
  elif isinstance(path,Path):
    dir_path=path.resolve()
  else:
    raise errors.InvalidInputError("path must be str or Path object")

  if not dir_path.is_dir():
    raise errors.InvalidInputError("path is not directory")

  tensors=[img_path_to_tensor(i) for i in dir_path.iterdir() if i.suffix.endswith((".png",".jpg"))]
  dir_max_width=max([t.shape[1] for t in tensors])
  dir_max_height=max([t.shape[2] for t in tensors])
  transform=v2.Compose([v2.ToImage(),v2.Resize(size=(dir_max_width,dir_max_height)),v2.ToDtype(torch.float32,scale=True)])
  r_tensors=[transform(t) for t in tensors]
  batch_list = [x.unsqueeze(0) for x in r_tensors]
  return torch.cat(batch_list)
