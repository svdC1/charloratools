from unittest.mock import patch, MagicMock
from charloratools.SysFileManager import ImgManager
import pytest

@patch('charloratools.SysFileManager.Image.open')  # Mock image opening
def test_img_manager_basic_operations(mock_image_open):
    # Mock the opened image object with width and height properties
    mock_image = MagicMock()
    mock_image.width = 800
    mock_image.height = 600
    mock_image_open.return_value = mock_image
    
    img_manager = ImgManager("tests/fixtures/sample_image.jpg")
    assert img_manager.width == 800
    assert img_manager.height == 600
    mock_image_open.assert_called_once_with("tests/fixtures/sample_image.jpg")

@patch('charloratools.SysFileManager.Image.open')  # Mock image opening
@patch('charloratools.SysFileManager.Image.save')  # Mock image saving
def test_img_resize(mock_image_save, mock_image_open):
    # Mock the opened image object
    mock_image = MagicMock()
    mock_image.width = 800
    mock_image.height = 600
    mock_image.resize.return_value = MagicMock(width=200, height=200)  # Mock resize output
    mock_image_open.return_value = mock_image
    
    img_manager = ImgManager("tests/fixtures/sample_image.jpg")
    img_manager.resize(200, inplace=False, output_dir="tests/output")
    
    mock_image_open.assert_called_once_with("tests/fixtures/sample_image.jpg")
    mock_image.resize.assert_called_once_with((200, 200))
    mock_image_save.assert_called()  # Ensure save was called after resizing
