import pytest
import shutil
from PIL import Image
import numpy as np
import charloratools as clt
from contextlib import nullcontext
import tempfile
from pathlib import Path
from conftest import setup_dirs,create_random_image

temp_dir1,temp_dir2,temp_dir3,temp_dir4,temp_dir_paths,temp_dir1_imgs,temp_dir2_imgs = setup_dirs()

@pytest.mark.parametrize("opts,expected",[((temp_dir_paths[0],True,True,'sha256',False,False),([temp_dir_paths[0]],['im1.png','im2.png','im3.png'])),
                                           ((temp_dir_paths[1],True,True,'phash',False,False),([temp_dir_paths[1],['im4.png']])),
                                           ((temp_dir_paths[3]/"new_dir",True,False,None,True,False),None)])
def test_dirisvalid(opts,expected):
  path=opts[0]
  check_images=opts[1]
  return_info=opts[2]
  hashtype=opts[3]
  create_if_not_found=opts[4]
  show_tqdm=opts[5]
  result=clt.utils.dirisvalid(path,check_images,return_info,hashtype,create_if_not_found,show_tqdm)
  if expected: 
    assert result[0]==expected[0]
  else:
    assert result
