from PIL import Image
import numpy as np
import tempfile
import urllib.request
from pathlib import Path
import sys
import logging

sys.path.append('./src')

# Setup Logging

logging.basicConfig(level="INFO", style="{",
                    format="{name} - {levelname} - {message}")


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


def download_from_cloudfare(img_id: str,
                            path: str | Path,
                            fn: str,
                            size: str = '400x400',
                            **kwargs):

    if not isinstance(img_id, str):
        raise TypeError("img_id must be a str")

    if not isinstance(path, str) and not isinstance(path, Path):
        raise TypeError("path must be a str or Path-like object")

    if not isinstance(fn, str):
        raise TypeError("fn must be a str")

    if not isinstance(size, str):
        raise TypeError("size must be a str")

    if isinstance(path, str):
        try:
            path = Path(path).resolve()
            if not path.exists() or not path.is_dir():
                raise RuntimeError(f"{path} is not a dir or doesn't exist.")

        except Exception as e:
            raise RuntimeError(f"Couldn't resolve path {path} : {e}")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Chrome')]
    urllib.request.install_opener(opener)

    acc_hash = "YCQ3OFRYiR1R_AeUslNHiw"
    base_url = "https://imagedelivery.net"

    img_url = f"{base_url}/{acc_hash}/{img_id}/{size}"

    # Try to Download 3 times
    try:
        for _ in range(3):
            try:
                save_path = path / fn
                urllib.request.urlretrieve(img_url,
                                           save_path)
                d_msg = f"Download sucessful, saved to {save_path}"
                logging.info(d_msg)
                return save_path
            except Exception as e:
                logging.info(f"Download failed : {e}, Retrying...")

    except Exception as e:
        raise RuntimeError(f"Error downloading from cloudfare : {e}")


def download_test_images():
    temp_dir1 = tempfile.TemporaryDirectory()
    temp_dir2 = tempfile.TemporaryDirectory()
    temp_dir3 = tempfile.TemporaryDirectory()
    temp_dir4 = tempfile.TemporaryDirectory()
    temp_dir_paths = [Path(i.name).resolve()
                      for i in [temp_dir1, temp_dir2, temp_dir3, temp_dir4]]

    test_face1_id = "4cb492db-609f-4b30-20dd-f21c71a9a300"
    test_face2_id = "aecfc808-015d-4731-60d3-26a77f840f00"
    size = "400x400"

    download_from_cloudfare(test_face1_id,
                            temp_dir_paths[0],
                            "test_face1.jpg",
                            size)

    download_from_cloudfare(test_face2_id,
                            temp_dir_paths[0],
                            "test_face2.jpg",
                            size)

    download_from_cloudfare(test_face1_id,
                            temp_dir_paths[1],
                            "test_face1.jpg",
                            size)

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


def download_test_faces():
    tempdir = tempfile.TemporaryDirectory()

    tdir_path = Path(tempdir.name).resolve()

    img_info = [{"img_id": 'f096ae47-caf2-43c3-f665-650031569200',
                 "path": tdir_path,
                 "fn": "face1.jpg",
                 "size": '400x400'
                 },
                {"img_id": 'b298c1ac-3b72-45f4-248d-8f7d4b12a100',
                 "path": tdir_path,
                 "fn": "face2.jpg",
                 "size": '400x400'
                 },
                {"img_id": 'd86b8943-68c1-4f43-702f-db0633f98000',
                 "path": tdir_path,
                 "fn": "filler.jpg",
                 "size": '400x400'
                 }]

    for d in img_info:
        download_from_cloudfare(**d)
    return tempdir


def cleanup(tempdirs: list):
    for tempdir in tempdirs:
        tempdir.cleanup()
