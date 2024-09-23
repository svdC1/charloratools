from conftest import setup_dirs
import pytest
import charloratools as clt

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
       "return_info": True}, 1)],
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


def test_filter_ref():
    with pytest.raises(clt.errors.NoFaceDetectedInReferenceImage):
        ff = clt.FilterAI.FaceRecognizer(path=temp_dir_paths[0])
        ff.filter_images_without_specific_face(
            ff.gallery.image_paths[0], temp_dir_paths[1])
