from conftest import setup_dirs, create_random_image
import pytest
from pathlib import Path
import tempfile
from PIL import Image
import charloratools as clt
from charloratools.SysFileManager import ImgManager
from charloratools.errors import InvalidPathError

(temp_dir1, temp_dir2, temp_dir3, temp_dir4,
 temp_dir_paths, temp_dir1_imgs, temp_dir2_imgs) = setup_dirs()

hashtype_lst = ['sha256', 'phash', 'dhash', 'avg_hash']
tqdm_lst = [True, False]


@pytest.mark.parametrize("show_tqdm", tqdm_lst)
@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gallery_manager(hashtype, show_tqdm):
    gm = clt.SysFileManager.GalleryManager(
        path=temp_dir_paths[0], hashtype=hashtype,
        show_tqdm=show_tqdm)
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
        create_random_image('random', gm.path)
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
def test_gm_sub(hashtype: str):
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
    im1_1 = gm1[im1]
    im1_2 = gm1[im1.path]
    im1_3 = gm1[str(im1.path)]

    assert isinstance(im1, clt.SysFileManager.ImgManager)
    assert im1 == im1_1 == im1_2 == im1_3


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_gm_getitem_error(hashtype):
    gm1 = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                            hashtype=hashtype)

    with pytest.raises(InvalidPathError):
        gm1['im1.png']


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
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    gm1 = clt.SysFileManager.GalleryManager(path=t_path,
                                            hashtype=hashtype)
    gm1 += gm1[0]
    gm1.delete_duplicates()
    assert len(gm1) == 1


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_tmp_manager_invalid_output_dir(hashtype):
    invalid_dir = Path("/invalid/path")

    with pytest.raises(clt.errors.InvalidInputError):
        with clt.SysFileManager.TmpManager(hashtype=hashtype,
                                           output_dir=invalid_dir):
            pass


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_tmp_manager_non_image_files(hashtype):
    with clt.SysFileManager.TmpManager(hashtype=hashtype,
                                       output_dir=Path("/tmp")) as manager:
        non_image_file = manager.tmp_path / "test.txt"
        with open(non_image_file, "w") as f:
            f.write("This is a test file.")

        assert non_image_file.exists()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_tmp_manager_cleanup_on_exception(hashtype):
    try:
        with clt.SysFileManager.TmpManager(hashtype=hashtype,
                                           output_dir=Path("/tmp")) as manager:
            raise RuntimeError("Simulated error")
    except RuntimeError:
        pass

    assert not manager.is_open


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_invalid_image_path(hashtype):
    with pytest.raises(clt.errors.InvalidPathError):
        clt.SysFileManager.ImgManager("/invalid/path/to/image.jpg", hashtype)


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_unsupported_image_type(hashtype):
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    txt_file = t_path / 'random.txt'
    with open(txt_file, "w") as f:
        f.write("This is a text file.")

    with pytest.raises(clt.errors.ImageTypeNotSupportedError):
        clt.SysFileManager.ImgManager(txt_file, hashtype)


def test_unsupported_hash_type():
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    with pytest.raises(clt.errors.ImgHashNotSupportedError):
        imanager = ImgManager(t_path / "random.png")
        imanager.to_hash("unsupported_hash_type")


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_deleted_image_operations(hashtype):
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    imanager = clt.SysFileManager.ImgManager(t_path / 'random.png', hashtype)
    imanager.delete()
    with pytest.raises(clt.errors.ImageIsDeletedError):
        imanager.to_hash()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_img_copy_with_existing_file(hashtype):
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    imanager = clt.SysFileManager.ImgManager(t_path / 'random.png',
                                             hashtype)
    imanager.copy_to(temp_dir_paths[2])
    assert (temp_dir_paths[2] / 'random.png').exists()


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_resize_image_aspect_ratio(hashtype):
    gm = clt.SysFileManager.GalleryManager(path=temp_dir_paths[0],
                                           hashtype=hashtype)
    gm.resize_all(max_size=300, keep_aspect_ratio=True)
    with Image.open(gm.img_managers[0].path) as img:
        assert img.width == 300 or img.height == 300


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_tmp_manager_cleanup_on_permission_error(hashtype):
    with pytest.raises(PermissionError):
        with clt.SysFileManager.TmpManager(hashtype=hashtype):
            raise PermissionError("Simulated permission error")


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_delete_multiple_duplicates(hashtype):
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    gm1 = clt.SysFileManager.GalleryManager(path=t_path,
                                            hashtype=hashtype)
    gm1 += gm1[0]
    gm1 += gm1[0]
    gm1.delete_duplicates()
    assert len(gm1) == 1


@pytest.mark.parametrize("hashtype", hashtype_lst)
def test_refresh_decorator(hashtype):
    temp = tempfile.TemporaryDirectory()
    t_path = Path(temp.name).resolve()
    create_random_image('random', t_path)
    gm = clt.SysFileManager.GalleryManager(path=t_path,
                                           hashtype=hashtype)
    initial_count = len(gm)
    create_random_image('new_image', t_path)
    gm.resize_all(max_size=200)
    assert len(gm) == initial_count + 1
