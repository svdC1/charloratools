import pytest
import charloratools as clt
import torch
import pandas as pd
import tempfile
from conftest import setup_dirs, create_empty_txt_file
from conftest import download_test_faces
from conftest import download_from_cloudfare
from pathlib import Path
from charloratools.errors import InvalidInputError, InfoDictFormatError

(temp_dir1, temp_dir2, temp_dir3, temp_dir4,
 temp_dir_paths, temp_dir1_imgs, temp_dir2_imgs) = setup_dirs()


@pytest.mark.parametrize("opts,expected", [((temp_dir_paths[0], True, True,
                                             'sha256', False, False),
                                            ([temp_dir_paths[0], ['im1.png',
                                                                  'im2.png',
                                                                  'im3.png']])
                                            ),
                                           ((temp_dir_paths[1],
                                             True, True,
                                             'phash', False, False),
                                            ([temp_dir_paths[1], ['im4.png']])
                                            ),
                                           ((temp_dir_paths[3]/"new_dir", True,
                                             False, None, True, False), None)])
def test_dirisvalid(opts, expected):
    path = opts[0]
    check_images = opts[1]
    return_info = opts[2]
    hashtype = opts[3]
    create_if_not_found = opts[4]
    show_tqdm = opts[5]
    result = clt.utils.dirisvalid(
        path, check_images, return_info, hashtype,
        create_if_not_found, show_tqdm)
    if expected:
        assert result[0] == expected[0]
    else:
        assert result


def test_dir_to_tensor():
    t = clt.utils.dir_path_to_img_batch(temp_dir_paths[0])
    assert (isinstance(t, torch.Tensor) and t.shape[0] == 3)


@pytest.mark.parametrize("opts,expected",
                         [((temp_dir_paths[0] / "im1.png", None), 100),
                          ((temp_dir_paths[0] / "im1.png", (300, 300)), 300)
                          ])
def test_img_to_tensor(opts, expected):
    img_path = opts[0]
    nsize = opts[1]
    t = clt.utils.img_path_to_tensor(img_path, nsize)
    assert (isinstance(t, torch.Tensor) and t.shape[1] == expected)


def test_img_path_to_tensor_non_image():
    temp = tempfile.TemporaryDirectory()
    invalid_img_path = create_empty_txt_file("Test", temp.name)

    with pytest.raises(InvalidInputError):
        clt.utils.img_path_to_tensor(invalid_img_path)
    temp.cleanup()


def test_img_path_to_tensor_missing_file():
    missing_img_path = Path("missing_image.jpg")
    with pytest.raises(FileNotFoundError):
        clt.utils.img_path_to_tensor(missing_img_path)


def test_info_dict_2_pandas():
    tempdir = download_test_faces()
    temp_out = tempfile.TemporaryDirectory()
    t_out_path = Path(temp_out.name).resolve()
    tdir_path = Path(tempdir.name).resolve()

    # Test Type Error
    with pytest.raises(InfoDictFormatError):
        clt.utils.InfoDict2Pandas('Test')

    # Test from Filter Faces
    fr = clt.FilterAI.FaceRecognizer(tdir_path)
    result = fr.filter_images_without_face(output_dir=t_out_path,
                                           return_info=True)
    gm, info_dict = result

    # Test dict type
    to_pandas_result = clt.utils.InfoDict2Pandas(info_dict)

    # Test lst type
    to_pandas_result2 = clt.utils.InfoDict2Pandas(info_dict['info_dict_lst'])

    # Test split dict type
    split_df_test_result = clt.utils.split_matched(info_dict)

    # Test split lst type
    split_df_test_result2 = clt.utils.split_matched(info_dict['info_dict_lst'])

    # Cleanup Temp
    tempdir.cleanup()
    temp_out.cleanup()
    # InfoDictTest
    assert isinstance(to_pandas_result['info_df'], pd.DataFrame)
    assert isinstance(to_pandas_result2['info_df'], pd.DataFrame)
    # Split test
    assert isinstance(split_df_test_result, dict)
    assert 'info_df' in split_df_test_result.keys()
    assert isinstance(split_df_test_result2, dict)
    assert 'info_df' in split_df_test_result2.keys()


def test_info_dict_2_pandas_ref():
    tempdir = download_test_faces()
    temp_out = tempfile.TemporaryDirectory()
    t_out_path = Path(temp_out.name).resolve()
    tdir_path = Path(tempdir.name).resolve()
    face1 = tdir_path / 'face1.jpg'

    # Test Type Error
    with pytest.raises(InfoDictFormatError):
        clt.utils.InfoDict2Pandas('Test')

    # Test from Filter Faces
    fr = clt.FilterAI.FaceRecognizer(tdir_path)
    result = fr.filter_images_without_specific_face(ref_img_path=face1,
                                                    output_dir=t_out_path,
                                                    return_info=True)
    gm, info_dict = result

    # Test dict type
    to_pandas_result = clt.utils.InfoDict2Pandas(info_dict)

    # Test lst type
    to_pandas_result2 = clt.utils.InfoDict2Pandas(info_dict['info_dict_lst'])

    # Test split dict type
    split_df_test_result = clt.utils.split_matched(info_dict)

    # Test split lst type
    split_df_test_result2 = clt.utils.split_matched(info_dict['info_dict_lst'])

    # Cleanup Temp
    tempdir.cleanup()
    temp_out.cleanup()
    # InfoDictTest
    assert isinstance(to_pandas_result['info_df'], pd.DataFrame)
    assert isinstance(to_pandas_result2['info_df'], pd.DataFrame)
    # Split test
    assert isinstance(split_df_test_result, dict)
    assert 'info_df' in split_df_test_result.keys()
    assert 'matched_ref_df' in split_df_test_result.keys()
    assert isinstance(split_df_test_result2, dict)
    assert 'info_df' in split_df_test_result2.keys()
    assert 'matched_ref_df' in split_df_test_result2.keys()


def test_dir_path_to_img_batch():
    tempdir = download_test_faces()
    tdir_path = Path(tempdir.name).resolve()
    png_id = 'bfb07292-3f47-4b35-b659-32b728b6bf00'
    download_from_cloudfare(png_id,
                            tdir_path,
                            'test_mixed.png')
    tensor = clt.utils.dir_path_to_img_batch(tdir_path)
    tempdir.cleanup()
    assert isinstance(tensor, torch.Tensor)
