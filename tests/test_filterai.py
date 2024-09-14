from conftest import setup_dirs,create_random_image
import pytest
import shutil
from PIL import Image
import numpy as np
import charloratools as clt
from contextlib import nullcontext
import tempfile
import pandas as pd
from pathlib import Path

temp_dir1,temp_dir2,temp_dir3,temp_dir4,temp_dir_paths,temp_dir1_imgs,temp_dir2_imgs = setup_dirs()

def test_face_recognizer():
  ff=clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
  assert isinstance(ff,clt.FilterAI.FaceRecognizer)

def test_filter_without_face():
  ff=clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
  gm=ff.filter_images_without_face(output_dir=temp_dir_paths[1])
  assert len(gm)==1

def test_filter_multiple_face():
  ff=clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
  gm=ff.filter_images_with_multiple_faces(output_dir=temp_dir_paths[1],return_info=True)
  assert len(gm)==1

def test_filter_ref():
  with pytest.raises(clt.errors.NoFaceDetectedInReferenceImage):
    ff=clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
    gm=ff.filter_images_without_specific_face(ff.gallery.image_paths[0],temp_dir_paths[1])

def test_info_dict():
  ff=clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
  gm,info_d=ff.filter_images_without_face(temp_dir_paths[0],return_info=True)
  result=clt.utils.InfoDict2Pandas(info_d)
  assert isinstance(result,pd.DataFrame)
  
  