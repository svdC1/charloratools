from PIL import Image
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.append('./src')


def create_random_image(filename, path):
    imarray = np.random.rand(100, 100, 3) * 255
    im = Image.fromarray(imarray.astype('uint8'))
    im.save(path/f"{filename}.png")


def setup_dirs():
    temp_dir1 = tempfile.TemporaryDirectory()
    temp_dir2 = tempfile.TemporaryDirectory()
    temp_dir3 = tempfile.TemporaryDirectory()
    temp_dir4 = tempfile.TemporaryDirectory()
    temp_dir_paths = [Path(i.name).resolve()
                      for i in [temp_dir1, temp_dir2, temp_dir3, temp_dir4]]
    create_random_image('im1', temp_dir_paths[0])
    create_random_image('im2', temp_dir_paths[0])
    create_random_image('im3', temp_dir_paths[0])
    create_random_image('im4', temp_dir_paths[1])
    temp_dir1_imgs = ['im1.png', 'im2.png', 'im3.png']
    temp_dir2_imgs = ['im4.png']
    return [temp_dir1, temp_dir2, temp_dir3,
            temp_dir4, temp_dir_paths,
            temp_dir1_imgs, temp_dir2_imgs]
