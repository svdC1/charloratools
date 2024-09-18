from conftest import setup_dirs, create_random_image
import pytest
from PIL import Image
import charloratools as clt

(temp_dir1, temp_dir2, temp_dir3, temp_dir4,
 temp_dir_paths, temp_dir1_imgs, temp_dir2_imgs) = setup_dirs()


def test_gallery_manager():
    gm = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    assert isinstance(gm, clt.SysFileManager.GalleryManager)


def test_resize_img():
    gm = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    gm.resize_all(max_size=200, keep_aspect_ratio=False, size=(200, 200))
    with Image.open(gm.img_managers[0].path) as img:
        assert (img.width == 200 and img.height == 200)


def test_gm_equals():
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    gm2 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[1], hashtype='phash')
    assert not gm1 == gm2


def test_gm_len():
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    assert len(gm1) == 3


def test_img_gallery_html():
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    savepath = gm1.to_html_img_gallery(temp_dir_paths[2])
    assert savepath[0].exists()


def test_tmp_manager():
    with clt.SysFileManager.TmpManager('sha256',
                                       output_dir=temp_dir_paths[3]) as gm:
        create_random_image('random.png', gm.path)
        assert gm.path.exists()


def test_gm_add():
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype='sha256')
    gm2 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[1], hashtype='phash')
    gm3 = gm1+gm2
    assert (len(gm3) == 4 and gm3.path.exists())


def test_img_copy():
    create_random_image('random', temp_dir_paths[3])
    imanager = clt.SysFileManager.ImgManager(
        temp_dir_paths[3]/'random.png', 'sha256')
    imanager.copy_to(temp_dir_paths[2])
    assert (temp_dir_paths[2]/'random.png').exists()


def test_gm_sub():
    with pytest.raises(clt.errors.OperationResultsInEmptyDirectoryError):
        gm1 = clt.SysFileManager.GalleryManager(
            path=temp_dir_paths[0], hashtype='sha256')
        gm2 = clt.SysFileManager.GalleryManager(
            path=temp_dir_paths[0], hashtype='crop_resistant')
        gm3 = gm1+gm2
        gm3-gm1
