from conftest import setup_dirs, download_test_images, cleanup
from conftest import download_test_faces
import pytest
import tempfile
import charloratools as clt
from tempfile import TemporaryDirectory as tmp
from pathlib import Path

(temp_dir1, temp_dir2, temp_dir3,
 temp_dir4, temp_dir_paths, temp_dir1_imgs, temp_dir2_imgs) = setup_dirs()


def test_face_recognizer():
    ff = clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
    assert isinstance(ff, clt.FilterAI.FaceRecognizer)


@pytest.mark.parametrize(
    "test_input,expected",
    [({"tempdir1": temp_dir_paths[0],
       "tempdir2": temp_dir_paths[1],
       "min_face_size": 20,
       "prob_threshold": None,
       "return_info": False}, 1),
     ({"tempdir1": temp_dir_paths[0],
       "tempdir2": temp_dir_paths[1],
       "min_face_size": 20,
       "prob_threshold": 0.9,
       "return_info": True}, 1),
     ({"tempdir1": temp_dir_paths[0],
       "tempdir2": temp_dir_paths[1],
       "min_face_size": 20,
       "prob_threshold": None,
       "return_info": True}, 1),
     ({"tempdir1": temp_dir_paths[0],
       "tempdir2": temp_dir_paths[1],
       "min_face_size": 20,
       "prob_threshold": None,
       "return_info": False}, 1)]
)
def test_filter_without_face(test_input,
                             expected):
    tdir1 = test_input['tempdir1']
    tdir2 = test_input['tempdir2']
    mfs = test_input['min_face_size']
    return_info = test_input['return_info']
    pb = test_input['prob_threshold']
    ff = clt.FilterAI.FaceRecognizer(path=tdir1)

    if not return_info:
        gm = ff.filter_images_without_face(output_dir=tdir2,
                                           min_face_size=mfs,
                                           return_info=return_info,
                                           prob_threshold=pb)
        assert len(gm) == expected
    else:
        result = ff.filter_images_without_face(output_dir=tdir2,
                                               min_face_size=mfs,
                                               return_info=return_info,
                                               prob_threshold=pb)
        gm = result[0]
        return_info = result[1]
        assert (len(gm) == expected and isinstance(return_info, dict))


def test_filter_multiple_face():
    ff = clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
    gm = ff.filter_images_with_multiple_faces(
        output_dir=temp_dir_paths[1], return_info=True)
    assert len(gm) == 2


def test_filter_ref_error():
    with pytest.raises(clt.errors.NoFaceDetectedInReferenceImage):
        ff = clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
        ff.filter_images_without_specific_face(
            ff.gallery.image_paths[0], temp_dir_paths[1])


def test_filter_ref_matches():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_without_specific_face(
        ff.gallery.image_paths[0], face_temp_dir_paths[2]
    )
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    assert gm_len == 1


def test_filter_ref_matches_prob():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_without_specific_face(
        ff.gallery.image_paths[0], face_temp_dir_paths[2],
        prob_threshold=0.9
    )
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    assert gm_len == 1


def test_filter_ref_no_match():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    clean_tmp = tmp()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_without_specific_face(
        ff.gallery[1].path,
        Path(clean_tmp.name).resolve())
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    clean_tmp.cleanup()

    assert gm_len == 1


def test_filter_ref_no_match_prob():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    clean_tmp = tmp()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_without_specific_face(
        ff.gallery[1].path,
        Path(clean_tmp.name).resolve(),
        prob_threshold=0.9)
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    clean_tmp.cleanup()

    assert gm_len == 1


def test_multiple_face():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    clean_tmp = tmp()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_with_multiple_faces(Path(clean_tmp.name).resolve())
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    clean_tmp.cleanup()

    assert gm_len == 2


def test_multiple_face_prob():
    (face_temp_dir1, face_temp_dir2, face_temp_dir3,
     face_temp_dir4, face_temp_dir_paths) = download_test_images()
    clean_tmp = tmp()
    ff = clt.FilterAI.FaceRecognizer(path=face_temp_dir_paths[0])
    gm = ff.filter_images_with_multiple_faces(Path(clean_tmp.name).resolve(),
                                              prob_threshold=0.9)
    gm_len = len(gm)
    cleanup([face_temp_dir1, face_temp_dir2, face_temp_dir3,
             face_temp_dir4])
    clean_tmp.cleanup()

    assert gm_len == 2


def test_save_images_with_detection_box():
    tempdir = download_test_faces()
    temp_out = tempfile.TemporaryDirectory()
    temp_out_db = tempfile.TemporaryDirectory()
    t_out_path = Path(temp_out.name).resolve()
    tdb_out_path = Path(temp_out_db.name).resolve()
    tdir_path = Path(tempdir.name).resolve()

    fr = clt.FilterAI.FaceRecognizer(tdir_path)
    gm, info = fr.filter_images_without_face(output_dir=t_out_path,
                                             return_info=True)
    idl = info['info_dict_lst']
    gm_db = fr.save_images_with_detection_box(info_dict_lst=idl,
                                              output_dir=tdb_out_path)
    gm_db_len = len(gm_db)
    tempdir.cleanup
    temp_out.cleanup
    temp_out_db.cleanup()
    assert gm_db_len == 3
