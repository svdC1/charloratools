import logging
import shutil
from pathlib import Path
import base64
# External Libs
import PIL
from PIL import Image
from tqdm import tqdm
import numpy as np
import imagehash
import hashlib
import tempfile
# Scripts
from . import errors
from . import utils


class ImgManager:
    """
    Handles image files for easier validation,comparison and processing when
    using OpenCV or PIL
    To compare images by numpy array correspondence instead of hashes use
    method 'arr_equal',
    operator defaults to specified hash comparison
    Args:
      path= string path to image file
      hashtype= Which type of hashing to use for image comparisons,defaults to
      'sha256'
                supports 'imagehashes' lib hashes
                ['phash','dhash,'avg_hash','crop_resistant']
    """
    OPERATION_ON_DELETED = "Trying to perform operation on deleted image"
    logger = logging.getLogger(__name__)

    def __init__(self, path: str | Path, hashtype: str = 'sha256'):
        if isinstance(path, str):
            path = Path(path).resolve()
        elif isinstance(path, Path):
            path = path
        # Checking if image path is valid
        if not path.exists():
            raise errors.InvalidPathError("Path doesn't exist")
        if not path.is_file():
            raise errors.InvalidInputError(
                "Image path does not exist or is not a file!")
        # Cheking if image extension is supported
        if not path.suffix.endswith(('.png',
                                     '.jpg',
                                     '.jpeg',
                                     '.tiff',
                                     '.bmp',
                                     '.gif')):
            raise errors.ImageTypeNotSupportedError(
                'Image type is not supported!')
        else:
            self.ext = path.suffix
        # Checking if pillow can open the image
        try:
            with Image.open(path) as img:
                self.array = np.array(img)
                self.dim = img.size
        except PIL.UnidentifiedImageError:
            raise errors.ImageIsUnopenableError(
                "Pillow can't open the image,check path")
        # End Validation
        self.path = path
        self.basename = path.stem+self.ext
        self.fname = path.stem
        self.hashtype = hashtype
        self.deleted = False
        self.hash = self.to_hash(hashtype)
        self.width = self.dim[0]
        self.height = self.dim[1]

    def __str__(self):
        s = f"""Source:{self.path}
                Size:{self.dim[0]}x{self.dim[1]};
                Deleted:{self.deleted};
                Hash:{str(self.hash)}"""
        return s

    def __repr__(self):
        return f"ImgManager(path={self.path},hashtype={self.hashtype})"

    def __bool__(self):
        if not self.deleted:
            return True
        else:
            return False

    def __eq__(self, other):

        return self.hash == other.hash

    def delete(self):
        """
        Deletes image from directory
        """
        if not self:
            raise errors.ImageIsDeletedError(self.OPERATION_ON_DELETED)
        try:
            self.path.unlink()
            self.logger.debug(f"Deleted Image {self.path}")
            self.deleted = True
        except Exception as e:
            self.deleted = False
            raise errors.ImgDeleteError(f"Failed to delete image : {str(e)}")

    def to_html_base64(self, no_html_tag=False):
        """
        Reads image in binary mode, encodes to base64
        and returns an html img tag
        """
        if not self:
            raise errors.ImageIsDeletedError(self.OPERATION_ON_DELETED)
        try:
            with open(self.path, 'rb') as binary_img:
                base64_img = base64.b64encode(binary_img.read())
            data_uri = base64_img.decode('utf-8')
            if no_html_tag:
                return base64_img
            return f'<img src="data:image/jpeg;base64,{data_uri}"/>'
        except Exception as e:
            raise errors.ImgOperationError(
                f"Failed to convert image to base64: {str(e)}")

    def arr_equals(self, other: np.ndarray):
        """
        Wrapper for 'np.array_equal(self.array,other)' where other
        is a numpy array
        Args:
          other: np.ndarray to be compared
        """
        if not isinstance(other, np.ndarray):
            raise errors.InvalidInputError("other must be numpy array")
        return np.array_equal(self.array, other)

    def copy_to(self,
                path: str | Path,
                name: str | None = None,
                copy_delete: bool = False):
        """
        Copies image to a new directory,appends a datetime.now unique suffix
        if a file of the same name already exists.
        """
        if not self:
            raise errors.ImageIsDeletedError(self.OPERATION_ON_DELETED)
        # Check dir
        if isinstance(path, str):
            path = Path(path).resolve()
        elif isinstance(path, Path):
            path = path.resolve()
        else:
            raise errors.InvalidInputError(
                "path must be string or Path object")
        if not path.is_dir():
            raise errors.InvalidInputError("Path must be a directory")
        if not path.exists():
            raise errors.InvalidPathError("Path Doesn't exist")

        if name:
            if not isinstance(name, str):
                raise errors.InvalidInputError("Name must be str")

            if copy_delete:
                try:
                    (path/name).unlink()
                except Exception as e:
                    raise errors.ImgOperationError(
                        f"Couldn't delete original image {path/name}:{str(e)}")
                self.logger.debug(f"Deleted Image {path/name}")

                copy_path = path/name

            else:
                copy_path = path/name

        elif self.basename in [i.stem+i.suffix for i in path.iterdir()]:
            copy_path = path / \
                f"{self.fname}_{utils.GetUniqueDtStr()}.{self.ext}"
        else:
            copy_path = path/self.basename
        if copy_path.exists():
            self.logger.warning("Image will be overwritten")
        try:
            shutil.copyfile(self.path, copy_path)
            self.logger.debug(f"Copied Image {self.path} to {copy_path}")
        except Exception as e:
            raise errors.ImgOperationError(
                f"Failed to copy image to {copy_path}: {str(e)}")

    def resize(self,
               max_size: int,
               keep_aspect_ratio=True,
               size: tuple | None = None,
               inplace: bool = True,
               output_dir: str | Path | None = None):
        if not inplace and output_dir is None:
            raise errors.InvalidInputError(
                "output_dir must be provided if inplace=False")
        if not keep_aspect_ratio and size is None:
            raise errors.InvalidInputError(
                "size must be provided if keep_aspect_ratio=False")
        if isinstance(output_dir, str):
            output_dir = Path(output_dir).resolve()
        elif isinstance(output_dir, Path):
            output_dir = output_dir.resolve()
        try:
            with Image.open(self.path) as img:
                if keep_aspect_ratio:
                    aspect_ratio = img.width / img.height
                    if aspect_ratio > 1:
                        # Image is wider than it is tall
                        new_width = max_size
                        new_height = int(max_size / aspect_ratio)
                    else:
                        # Image is taller than it is wide
                        new_height = max_size
                        new_width = int(max_size * aspect_ratio)
                    img_resized = img.resize(
                        (new_width, new_height), Image.LANCZOS)
                else:
                    img_resized = img.resize(size, Image.LANCZOS)
                if inplace:
                    self.delete()
                    img_resized.save(self.path, quality=95)
                    rw = img_resized.width
                    rh = img_resized.height
                    self.logger.debug(
                        f"Resized Image {self.basename} to {rw}x{rh}")
                    self.deleted = False
                else:
                    fn = self.fname
                    rw = img_resized.width
                    rh = img_resized.height
                    img_resized.save(
                        output_dir/f"{fn}_{rw}_{rh}.{self.ext}",
                        quality=95)
        except Exception as e:
            raise errors.ImgOperationError(
                f"Failed to resize image : {str(e)}")

    def to_hash(self, hashtype: str = 'sha256'):
        if hashtype not in ['sha256', 'phash', 'crop_resistant', 'avg_hash',
                            'dhash']:
            raise errors.ImgHashNotSupportedError()
        if not self:
            raise errors.ImageIsDeletedError(self.OPERATION_ON_DELETED)
        if hashtype == 'sha256':
            with open(self.path, "rb") as f:
                hash = hashlib.sha256(f.read()).hexdigest()
        elif hashtype == 'phash':
            with Image.open(self.path) as img:
                hash = imagehash.phash(img)
        elif hashtype == 'crop_resistant':
            with Image.open(self.path) as img:
                hash = imagehash.crop_resistant_hash(img)
        elif hashtype == 'avg_hash':
            with Image.open(self.path) as img:
                hash = imagehash.average_hash(img)
        elif hashtype == 'dhash':
            with Image.open(self.path) as img:
                hash = imagehash.dhash(img)
        return hash

    def __hash__(self):
        if self.hashtype == "sha256":
            return int(self.hash, 16)
        else:
            return hash(self.hash)


def refresh_decorator(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        prev_img_count = len(self.images)
        self.path, self.images = utils.dirisvalid(
            self.path,
            return_info=True,
            hashtype=self.hashtype,
            show_tqdm=self.show_tqdm)
        self.logger.debug(
            f"Refreshed images {prev_img_count}->{len(self.images)}")
        result = func(*args, **kwargs)
        prev_img_count = len(self.images)
        self.path, self.images = utils.dirisvalid(
            self.path,
            return_info=True,
            hashtype=self.hashtype,
            show_tqdm=self.show_tqdm)
        self.basename = self.path.stem
        self.ext_dir = self.path.parent
        self.ext_dir_name = self.path.parent.stem
        self.img_managers = [v for v in self.images.values()]
        self.image_paths = [k for k in self.images.keys()]
        self.logger.debug(
            f"Refreshed images {prev_img_count}->{len(self.images)}")
        return result
    return wrapper


class GalleryManager:
    """
    Handles management and validation of a directory of
    images using the ImgManager class.
    supports operators with other instances and paths to facilitate management.
    Args:
      path: string path to gallery directory
      hashtype: Type of hash to use for comparing images,defaults to sha256
    """
    logger = logging.getLogger(__name__)

    def __init__(self, path: str | Path,
                 hashtype: str = 'sha256',
                 show_tqdm: bool = False):
        if isinstance(path, str):
            self.path = Path(path).resolve()
        elif isinstance(path, Path):
            self.path = path
        else:
            raise errors.InvalidPathError(
                "path must be string or path-like object")
        self.basename = self.path.stem
        self.ext_dir = self.path.parent
        self.ext_dir_name = self.path.parent.stem
        self.hashtype = hashtype
        self.show_tqdm = show_tqdm
        self.path, self.images = utils.dirisvalid(
            self.path,
            return_info=True,
            hashtype=self.hashtype,
            show_tqdm=self.show_tqdm)
        self.img_managers = [v for v in self.images.values()]
        self.image_paths = [k for k in self.images.keys()]

    @refresh_decorator
    def change_hashtype(self, hashtype):
        self.hashtype = hashtype

    @refresh_decorator
    def __str__(self):
        return f"{[i.__str__() for i in self]}"

    def __repr__(self):
        return f"GalleryManager(path={self.path})"

    @refresh_decorator
    def __len__(self):
        return len(self.images)

    @refresh_decorator
    def __getitem__(self, key):
        if isinstance(key, str):
            imgmanager = ImgManager(
                path=Path(key).resolve(), hashtype=self.hashtype)
            if imgmanager in self:
                index = self.get_img_manager_index(imgmanager)

                return self.img_managers[index]
            else:
                raise KeyError(
                    "Path isn't valid or Image doesn't match any in gallery")
        elif isinstance(key, Path):
            imgmanager = ImgManager(path=key, hashtype=self.hashtype)
            if imgmanager in self:
                index = self.get_img_manager_index(imgmanager)
                return self.img_managers[index]
            else:
                raise KeyError(
                    "Path isn't valid or Image doesn't match any in gallery")

        elif isinstance(key, ImgManager):
            if key in self:
                index = self.get_img_manager_index(key)
                return self.img_managers[index]
            else:
                raise KeyError(
                    "Image doesn't match any image present in gallery")

        elif isinstance(key, int):
            return self.img_managers[key]
        else:
            ke = f"""Key must be a string with the image file name,
                     path-like object of image path,or ImgManager object,
                     got {type(key)}"""
            raise KeyError(ke)

    @refresh_decorator
    def __setitem__(self, key, value):
        if isinstance(value, str):
            oldimgmanager = self[key]
            nimg = ImgManager(Path(value).resolve(), hashtype=self.hashtype)
            nimg.copy_to(self.path, name=oldimgmanager.basename,
                         copy_delete=True)

        elif isinstance(value, ImgManager):
            value.copy_to(self.path)

        elif isinstance(value, Path):
            nimg = ImgManager(path=value.resolve(), hashtype=self.hashtype)
            nimg.copy_to(self.path)
        else:
            raise errors.OperationNotSupportedError("Operation not supported")

    @refresh_decorator
    def __iter__(self):
        return iter(tuple(self.img_managers))

    @refresh_decorator
    def __contains__(self, item):
        if isinstance(item, self.__class__):
            if self.hashtype != item.hashtype:
                item.change_hashtype(self.hashtype)
            shashes = [i.hash for i in self]
            ohashes = [i.hash for i in item]
            if self.hashtype == 'sha256':
                return np.isin(np.array(shashes), np.array(ohashes)).all()
            else:
                isin_bools = []
                for shash in shashes:
                    isin_bools.append(
                        np.any(np.array([shash == ohash for ohash in ohashes])
                               ))
                return np.all(isin_bools)

        elif isinstance(item, list):
            ohashes = []
            for i in item:
                if isinstance(i, ImgManager):
                    if i.hashtype != self.hashtype:
                        ohashes.append(i.to_hash(self.hashtype))
                    else:
                        ohashes.append(i.hash)
                elif isinstance(i, str):
                    ohashes.append(ImgManager(
                        path=i, hashtype=self.hashtype).hash)
            shashes = [i.hash for i in self]
            if self.hashtype == 'sha256':
                return np.isin(np.array(shashes), np.array(ohashes)).all()
            else:
                isin_bools = []
                for shash in shashes:
                    isin_bools.append(
                        np.any(np.array([shash == ohash for ohash in ohashes])
                               ))
                return np.all(isin_bools)

        elif isinstance(item, ImgManager):
            if self.hashtype != item.hashtype:
                hash = item.to_hash(self.hashtype)
                if self.hashtype == 'sha256':
                    return hash in [i.hash for i in self]
                else:
                    shashes = [i.hash for i in self]
                    return np.any([shash == hash for shash in shashes])
            else:
                if self.hashtype == 'sha256':
                    return item.hash in [i.hash for i in self]
                else:
                    shashes = [i.hash for i in self]
                    return np.any([shash == item.hash for shash in shashes])

        elif isinstance(item, str):
            nhash = ImgManager(item, hashtype=self.hashtype).hash
            if self.hashtype == 'sha256':
                return nhash in [i.hash for i in self]
            else:
                shashes = [i.hash for i in self]
                return np.any([shash == nhash for shash in shashes])
        else:
            raise errors.OperatorNotSupportedError(
                f'Operation not supported for type {type(item)}')

    @refresh_decorator
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self) != len(other):
                return False
            if self.hashtype != other.hashtype:
                other.change_hashtype(self.hashtype)

            if self.hashtype == 'sha256':
                return np.array_equal(np.sort(np.array([i.hash for i in self])
                                              ),
                                      np.sort(np.array([i.hash for i in other])
                                              ))
            else:
                shashes_tup = [(str(i.hash), i.hash)
                               for i in self].sort(key=lambda t: t[0])
                ohashes_tup = [(str(i.hash), i.hash)
                               for i in other].sort(key=lambda t: t[0])
                return np.all([stup[1] == otup[1] for stup,
                               otup in zip(shashes_tup, ohashes_tup)])

        else:
            return False

    @refresh_decorator
    def __ne__(self, other):
        return not self.__eq__(other)

    @refresh_decorator
    def __gt__(self, other):
        return len(self) > len(other)

    @refresh_decorator
    def __lt__(self, other):
        return len(self) < len(other)

    @refresh_decorator
    def __add__(self, other):
        if isinstance(other, self.__class__):
            # Addings two instances
            # Creating new dir
            if other.hashtype != self.hashtype:
                other.change_hashtype(self.hashtype)
            s_base = self.basename
            nn = f"{s_base}_add_{other.basename}_{utils.GetUniqueDtStr()}"
            new_dir_name = Path(f"{self.ext_dir}") / Path(nn)
            new_dir_name = utils.dirisvalid(
                new_dir_name,
                create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in tqdm(self, desc='Copying 1st instance images'):
                imgmanager.copy_to(new_dir_name.resolve())
            for oimgmanager in tqdm(other, desc='Copying 2nd instance images'):
                oimgmanager.copy_to(new_dir_name.resolve())
            # Hashtype returned is the left instance's one
            return GalleryManager(new_dir_name.resolve(),
                                  hashtype=self.hashtype)
        # Subtracting instance from str
        elif isinstance(other, str):
            nimgmanager = ImgManager(other, hashtype=self.hashtype)
            fn = Path(f"{self.basename}_add_{nimgmanager.basename}")
            new_dir_name = Path(f"{self.ext_dir}")/fn
            new_dir_name = utils.dirisvalid(
                new_dir_name, create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in self:
                imgmanager.copy_to(new_dir_name.resolve())
            nimgmanager.copy_to(new_dir_name.resolve())
            return GalleryManager(new_dir_name.resolve(),
                                  hashtype=self.hashtype)
        elif isinstance(other, ImgManager):
            if other.hashtype != self.hashtype:
                other = ImgManager(other.path)
            new_dir_name = Path(f"{self.ext_dir}") / \
                Path(f"{self.basename}_add_{other.basename}")
            new_dir_name = utils.dirisvalid(
                new_dir_name, create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in self:
                imgmanager.copy_to(new_dir_name.resolve())
            other.copy_to(new_dir_name.resolve())
            return GalleryManager(new_dir_name.resolve(),
                                  hashtype=self.hashtype)
        else:
            raise errors.OperationNotSupportedError("Operation not supported")

    @refresh_decorator
    def __sub__(self, other):
        # Subtracting two instances
        if isinstance(other, self.__class__):
            if other.hashtype != self.hashtype:
                other.change_hashtype(self.hashtype)
            if self.hashtype == 'sha256':
                ohashes = [k.hash for k in other]
                imgs_to_add = [i for i in self if i.hash not in ohashes]
            else:
                imgs_to_add = []
                for simg in self:
                    if np.any([not (simg.hash == oimg.hash) for oimg in other]
                              ):
                        imgs_to_add.append(simg)

            if len(imgs_to_add) == 0:
                raise errors.OperationResultsInEmptyDirectoryError(
                    "Operation would result in empty directory")
            new_dir_name = Path(f"{self.ext_dir}") / \
                Path(f"{self.basename}_sub_{other.basename}")
            new_dir_name = utils.dirisvalid(
                new_dir_name, create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in tqdm(imgs_to_add, desc='Copying images'):
                imgmanager.copy_to(new_dir_name.resolve())
            return GalleryManager(new_dir_name.resolve(),
                                  hashtype=self.hashtype)

        elif isinstance(other, str):
            nimgmanager = ImgManager(other, hashtype=self.hashtype)
            img_to_add = [i for i in self if i != nimgmanager]
            if len(img_to_add) == 0:
                raise errors.OperationResultsInEmptyDirectoryError(
                    "Operation would result in empty directory")
            fn = Path(f"{self.basename}_sub_{nimgmanager.basename}")
            new_dir_name = Path(
                f"{self.ext_dir}")/fn
            new_dir_name = utils.dirisvalid(
                new_dir_name, create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in img_to_add:
                imgmanager.copy_to(new_dir_name.resolve())
        elif isinstance(other, ImgManager):
            img_to_add = [i for i in self if i != other]
            if len(img_to_add) == 0:
                raise errors.OperationResultsInEmptyDirectoryError(
                    "Operation would result in empty directory")
            new_dir_name = Path(f"{self.ext_dir}") / \
                Path(f"{self.basename}_sub_{other.basename}")
            new_dir_name = utils.dirisvalid(
                new_dir_name, create_if_not_found=True,
                show_tqdm=self.show_tqdm)
            for imgmanager in img_to_add:
                imgmanager.copy_to(new_dir_name.resolve())
        else:
            raise errors.OperationNotSupportedError("Operation not supported")

    @refresh_decorator
    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            if other.hashtype != self.hashtype:
                other.change_hashtype(self.hashtype)
            for imgmanager in tqdm(other, desc='Copying Images'):
                imgmanager.copy_to(self.path)
            return self
        elif isinstance(other, str):
            nimgmanager = ImgManager(other, hashtype=self.hashtype)
            nimgmanager.copy_to(self.path)
            return self
        elif isinstance(other, ImgManager):
            other.copy_to(self.path)
            return self
        else:
            raise errors.OperationNotSupportedError("Operation not supported")

    @refresh_decorator
    def __isub__(self, other):
        if isinstance(other, self.__class__):
            if other.hashtype != self.hashtype:
                other.change_hashtype(self.hashtype)
            if other == self:
                raise errors.OperationResultsInEmptyDirectoryError(
                    "Operation would result in an empty directory")
            for imgmanager in tqdm(self, desc="Checking Hashes"):
                if imgmanager in other:
                    imgmanager.delete()
        elif isinstance(other, str):
            nimgmanager = ImgManager(other, hashtype=self.hashtype)
            for imgmanager in self:
                if imgmanager == nimgmanager:
                    imgmanager.delete()
        elif isinstance(other, ImgManager):
            for imgmanager in self:
                if imgmanager == other:
                    imgmanager.delete()
            return self
        else:
            raise errors.OperatorNotSupportedError(
                f'Operation not supported for type {type(other)}')

    @refresh_decorator
    def __hash__(self):
        img_hashes = tuple([hash(manager) for manager in self.img_managers])
        return hash(img_hashes)

    @refresh_decorator
    def to_html_img_gallery(self, output_dir: str,
                            separate_elements: bool = False):
        """
        Method to create a standalone HTML image gallery for preview from the
        instance's current directory
        Creates a generated_img_gallery.html in output_dir
        :param str output_dir Directory to create the .html file in
        :param bool separate_elements: Return only the image gallery html div
        and necessary <head> and <body> elements
        instead of a fully formatted website
        (For html injection purposes such as Django Templates),
        defaults to False
        """
        output_dir = utils.dirisvalid(
            output_dir, create_if_not_found=True, show_tqdm=self.show_tqdm)
        website_template = """<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Preview Image Gallery</title>
        <link rel="icon"
         href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.6.0/svgs/solid/images.svg"/>
        <script
         src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js">
        </script>
        <link
         href="https://cdnjs.cloudflare.com/ajax/libs/fotorama/4.6.4/fotorama.css"
         rel="stylesheet">
        <script
         src="https://cdnjs.cloudflare.com/ajax/libs/fotorama/4.6.4/fotorama.js"></script>
        <link
         href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet"
        integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
         crossorigin="anonymous">
        </head><body style="background-color:#024950;">
        <div class="container text-center"><div class="row">
        <div class="col"><h1 style="color:#AFDDE5">Image Gallery</h1></div>
        </div>
        <div class="row"><div class="fotorama col"
         data-allowfullscreen="native"
         data-width="100%" data-ratio="4/3">{0}</div></div></div>
         <div class="container text-center">
        <p style="color:#AFDDE5">
        Photo Gallery Credits:<a href="https://fotorama.io/">Fotoroma</a></p>
        </div>
        <script
         src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
         integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
          crossorigin="anonymous"></script>
        </body></html>
        """
        body_template = """<div class="container text-center"><div class="row">
                     <div class="fotorama col" data-allowfullscreen="native"
                      data-width="100%" data-ratio="4/3">{0}</div>
                     </div></div>
                     <script
                      src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
                      integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
                       crossorigin="anonymous"></script>
                     """
        head_template = """<meta charset="utf-8">
                     <meta name="viewport" content="width=device-width,
                      initial-scale=1">
                     <script
                      src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
                     <link
                      href="https://cdnjs.cloudflare.com/ajax/libs/fotorama/4.6.4/fotorama.css"
                       rel="stylesheet">
                     <script
                     src="https://cdnjs.cloudflare.com/ajax/libs/fotorama/4.6.4/fotorama.js"></script>
                     <link
                      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
                      rel="stylesheet"
                      integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
                      crossorigin="anonymous">
                   """

        html_tags = ''
        for i in self:
            img_tag = i.to_html_base64()
            html_tags += img_tag
        template_with_img = website_template.format(html_tags)
        body_template_with_img = body_template.format(html_tags)
        if separate_elements:
            self.logger.info("html body formatted sucessfully")
            return head_template, body_template_with_img
        else:
            save_path = output_dir / \
                f'generated_img_gallery_{utils.GetUniqueDtStr()}.html'
            with open(save_path, 'w') as f:
                f.write(template_with_img)
            self.logger.info('.html image gallery saved sucessfully')
            return save_path, template_with_img

    @refresh_decorator
    def delete_duplicates(self):
        seen = []
        dups = []
        for i in self:
            if i in seen:
                dups.append(i)
            seen.append(i)
        for i in dups:
            i.delete()

    @refresh_decorator
    def resize_all(self,
                   max_size: int,
                   keep_aspect_ratio=True,
                   size: tuple | None = None,
                   inplace: bool = True,
                   output_dir: str | Path | None = None):
        if not inplace:
            output_dir = utils.dirisvalid(
                output_dir, create_if_not_found=True, show_tqdm=self.show_tqdm)
            if keep_aspect_ratio:
                for i in self:
                    i.resize(max_size, True, None, False, output_dir)
                return GalleryManager(path=output_dir, hashtype=self.hashtype)

            else:
                for i in self:
                    i.resize(max_size, False, size, False, output_dir)
                return GalleryManager(path=output_dir, hashtype=self.hashtype)

        else:
            if keep_aspect_ratio:
                for i in self:
                    i.resize(max_size, True, None, True, None)
                return self
            else:
                for i in self:
                    i.resize(max_size, False, size, True, None)
                return self

    @refresh_decorator
    def get_img_manager_index(self, img_manager: ImgManager):
        if img_manager not in self:
            raise KeyError("Image isn't in gallery")
        else:
            for i, smanager in enumerate(self.img_managers):
                if img_manager == smanager:
                    return i


class TmpManager(GalleryManager):
    """
    Handles creation and deletion of temporary directory for image operations
    Extends GalleryManager to use a temporary directory.
    """

    def __init__(self,
                 hashtype: str,
                 save_content_on_deletion: bool = False,
                 output_dir: str | Path | None = None):
        # initializing gallery manager attrs
        if output_dir is None:
            output_dir = None
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir).resolve()
        elif isinstance(output_dir, Path):
            output_dir = output_dir.resolve()
        else:
            raise errors.InvalidInputError(
                "output_dir must be str or path-like object")
        self.prev_tmp_dir = None
        self.path = None
        self.basename = None
        self.ext_dir = None
        self.ext_dir_name = None
        self.show_tqdm = None
        self.images = None
        self.img_managers = None
        self.image_paths = None
        # --
        self.prev_tempdir_var = None
        self.temp_dir = None
        self.tmp_path = None
        self.hashtype = hashtype
        self.save_content_on_deletion = save_content_on_deletion
        self.output_dir = Path(output_dir).resolve(
        ) if output_dir else self.tmp_path
        self.is_open = False

    def __str__(self):
        s = f"""Temporary Directory;
        Name:{self.tmp_path};
        Opened:{self.is_open}"""
        return s

    def __repr__(self):
        ht = f"hashtype={self.hashtype}"
        sc = f"save_content_on_deletion={self.save_content_on_deletion}"
        return f"TmpManager({ht},{sc},output_dir={self.output_dir})"

    def __enter__(self):
        self.prev_tempdir_var = tempfile.tempdir
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name).resolve()
        logging.debug(f"Created tmp dir at {self.tmp_path}")
        placeholder = Image.fromarray(np.reshape(
            np.arange(0, 100, 1, dtype=np.uint8), (10, 10)))
        placeholder.save(self.tmp_path / 'placeholder.jpg')
        super().__init__(path=self.tmp_path, hashtype=self.hashtype)
        self.is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Transfer contents to a permanent location if needed
        if self.save_content_on_deletion and len(self) != 0:
            for img in self:
                if img.basename != "placeholder.jpg":
                    img.copy_to(self.output_dir)
        self.temp_dir.cleanup()
        self.is_open = False

    def __del__(self):
        # Ensure cleanup if not already done
        self.temp_dir.cleanup()
