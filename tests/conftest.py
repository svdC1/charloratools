from PIL import Image
import numpy as np
import tempfile
import urllib.request
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


def download_test_images():
    temp_dir1 = tempfile.TemporaryDirectory()
    temp_dir2 = tempfile.TemporaryDirectory()
    temp_dir3 = tempfile.TemporaryDirectory()
    temp_dir4 = tempfile.TemporaryDirectory()
    temp_dir_paths = [Path(i.name).resolve()
                      for i in [temp_dir1, temp_dir2, temp_dir3, temp_dir4]]
    cloudfare_url = "https://imagedelivery.net"
    cloudfare_id = "YCQ3OFRYiR1R_AeUslNHiw"
    test_face1_id = "4cb492db-609f-4b30-20dd-f21c71a9a300"
    test_face2_id = "aecfc808-015d-4731-60d3-26a77f840f00"
    size = "400x400"
    test_face1_url = f"{cloudfare_url}/{cloudfare_id}/{test_face1_id}/{size}"
    test_face2_url = f"{cloudfare_url}/{cloudfare_id}/{test_face2_id}/{size}"
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Chrome')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(test_face1_url,
                               temp_dir_paths[0] / "test_face1.jpg")
    urllib.request.urlretrieve(test_face2_url,
                               temp_dir_paths[0] / "test_face2.jpg")
    urllib.request.urlretrieve(test_face1_url,
                               temp_dir_paths[1] / "test_face1.jpg")
    return [temp_dir1, temp_dir2, temp_dir3, temp_dir4, temp_dir_paths]


def create_empty_txt_file(fname, dir_):
    txt_path = Path(dir_).resolve() / Path(f"{fname}.txt")
    with open(txt_path, 'w') as n_txt:
        n_txt.write("Test")
    return txt_path


def setup_invalid_dirs():
    temp_dir_txt = tempfile.TemporaryDirectory()
    create_empty_txt_file("Test", temp_dir_txt.name)
    temp_dir_paths = [Path(i.name) for i in [temp_dir_txt]]
    return temp_dir_paths


def cleanup(tempdirs: list):
    for tempdir in tempdirs:
        tempdir.cleanup()
