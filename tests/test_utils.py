from unittest.mock import patch
from charloratools.utils import dirisvalid, GetUniqueDtStr
from pathlib import Path

@patch.objects(Path,'exists')  # Mock file existence check
@patch.object(Path,'mkdir')     # Mock directory creation
def test_create_new_directory(mock_mkdir, mock_exists):
    # Mock Path.exists to return False (directory doesn't exist)
    mock_exists.return_value = False
    
    path = dirisvalid("tests/new_directory", create_if_not_found=True)
    
    mock_mkdir.assert_called_once_with("tests/new_directory")
    mock_exists.assert_called_once_with(parents=True,exist_ok=True)

def test_get_unique_dt_str():
    dt_str = GetUniqueDtStr()
    assert len(dt_str) > 0  # Ensure that a string is returned
