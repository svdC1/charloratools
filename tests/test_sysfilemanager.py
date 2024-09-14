from conftest import setup_dirs,create_random_image
import pytest
import shutil
from PIL import Image
import numpy as np
import charloratools as clt
from contextlib import nullcontext
import tempfile
from pathlib import Path

temp_dir1,temp_dir2,temp_dir3,temp_dir4,temp_dir_paths,temp_dir1_imgs,temp_dir2_imgs = setup_dirs()

def test_gallery_manager():
  gm=clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],hashtype='sha256')
  assert isinstance(gm,clt.SysFileManager.GalleryManager)

def test_resize_img():
  gm=clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],hashtype='sha256')
  gm.resize_all(max_size=200,keep_aspect_ratio=False,size=(200,200))
  with Image.open(gm.img_managers[0].path) as img:
    assert (img.width==200 and img.height==200)
  