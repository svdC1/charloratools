from unittest.mock import patch, MagicMock
from charloratools.FilterAI import FaceRecognizer

@patch('charloratools.FilterAI.FaceRecognizer.filter_images_without_face')
def test_filter_images_without_face(mock_detect_faces):
    # Mock face detection to return an empty list (no faces detected)
    mock_detect_faces.return_value = []
    
    recognizer = FaceRecognizer("tests/fixtures/valid_images_dir")
    output_gallery = recognizer.filter_images_without_face("tests/output", min_face_size=20)
    
    assert len(output_gallery) == 0  # No images should be returned if no faces are detected
    mock_detect_faces.assert_called()

@patch('charloratools.FilterAI.FaceRecognizer.images_without_face')
def test_filter_images_with_faces(mock_detect_faces):
    # Mock face detection to return a face (simulate detecting faces)
    mock_face = MagicMock()
    mock_face.rect = (10, 10, 100, 100)  # Simulate bounding box of the face
    mock_detect_faces.return_value = [mock_face]
    
    recognizer = FaceRecognizer("tests/fixtures/valid_images_dir")
    output_gallery = recognizer.filter_images_without_face("tests/output", min_face_size=20)
    
    assert len(output_gallery) > 0  # Ensure images with faces are processed
    mock_detect_faces.assert_called()
