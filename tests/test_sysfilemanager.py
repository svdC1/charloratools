from conftest import setup_dirs, create_random_image
import pytest
from pathlib import Path
import tempfile
from PIL import Image
import charloratools as clt

(temp_dir1, temp_dir2, temp_dir3, temp_dir4,
 temp_dir_paths, temp_dir1_imgs, temp_dir2_imgs) = setup_dirs()

hashtype_lst = ['sha256', 'phash', 'dhash', 'avg_hash']


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gallery_manager(hashtype):
    gm = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    assert isinstance(gm, clt.SysFileManager.GalleryManager)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_resize_img(hashtype):
    gm = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    gm.resize_all(max_size=200, keep_aspect_ratio=False, size=(200, 200))
    with Image.open(gm.img_managers[0].path) as img:
        assert (img.width == 200 and img.height == 200)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_equals(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    gm2 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[1], hashtype='phash')
    assert not gm1 == gm2


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_len(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    assert len(gm1) == 3


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_img_gallery_html(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    savepath = gm1.to_html_img_gallery(temp_dir_paths[2])
    assert savepath[0].exists()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_tmp_manager(hashtype):
    with clt.SysFileManager.TmpManager(hashtype,
                                       output_dir=temp_dir_paths[3]) as gm:
        create_random_image('random.png', gm.path)
        assert gm.path.exists()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_add(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype)
    gm2 = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[1], hashtype='phash')
    gm3 = gm1+gm2
    assert (len(gm3) == 4 and gm3.path.exists())


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_img_copy(hashtype):
    create_random_image('random', temp_dir_paths[3])
    imanager = clt.SysFileManager.ImgManager(
        temp_dir_paths[3]/'random.png', hashtype)
    imanager.copy_to(temp_dir_paths[2])
    assert (temp_dir_paths[2]/'random.png').exists()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_sub(hashtype):
    with pytest.raises(clt.errors.OperationResultsInEmptyDirectoryError):
        gm1 = clt.SysFileManager.GalleryManager(
            path=temp_dir_paths[0], hashtype='sha256')
        gm2 = clt.SysFileManager.GalleryManager(
            path=temp_dir_paths[0], hashtype=hashtype)
        gm3 = gm1+gm2
        gm3-gm1


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_getitem(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                            hashtype=hashtype)
    im1 = gm1[0]
    assert isinstance(im1, clt.SysFileManager.ImgManager)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_set_item(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                            hashtype=hashtype)
    ntdir = tempfile.TemporaryDirectory()
    ntdir_path = Path(ntdir.name).resolve()
    create_random_image("im5", ntdir_path)
    impath = ntdir_path / 'im5.png'
    gm1[0] = impath
    imgmanager = clt.SysFileManager.ImgManager(path=impath,
                                               hashtype=hashtype)

    assert (imgmanager in gm1)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_ops(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                            hashtype=hashtype)
    gm2 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[1],
                                            hashtype='sha256')
    lc = gm1 > gm2
    lf = gm1 < gm2
    ne_c = gm1 != gm2
    gm1 += gm2
    gm1 -= gm2
    assert (lc is True) and (lf is False) and (ne_c is True)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_duplicates(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                            hashtype=hashtype)
    gm1 += gm1[0]
    gm1.delete_duplicates()
    assert len(gm1) == 7


def test_tmp_manager_invalid_output_dir():
    invalid_dir = Path("/invalid/path")

    with pytest.raises(clt.errors.InvalidInputError):
        with clt.SysFileManager.TmpManager(hashtype='sha256',
                                           output_dir=invalid_dir):
            pass


def test_tmp_manager_non_image_files():
    with clt.SysFileManager.TmpManager(hashtype='sha256',
                                       output_dir=Path("/tmp")) as manager:
        non_image_file = manager.tmp_path / "test.txt"
        with open(non_image_file, "w") as f:
            f.write("This is a test file.")

        assert non_image_file.exists()


def test_tmp_manager_cleanup_on_exception():
    try:
        with clt.SysFileManager.TmpManager(hashtype='sha256',
                                           output_dir=Path("/tmp")) as manager:
            raise RuntimeError("Simulated error")
    except RuntimeError:
        pass

    assert not manager.is_open
