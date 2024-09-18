import pytest
import charloratools as clt
import torch
from conftest import setup_dirs

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
