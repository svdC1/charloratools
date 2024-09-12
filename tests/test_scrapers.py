import pytest
from unittest.mock import patch, MagicMock
from charloratools.Scrapers import VSCOScraper
from pathlib import Path
import logging

@patch('charloratools.Scrapers.webdriver.Chrome')  # Mock the Selenium Chrome WebDriver
@patch('builtins.open', new_callable=MagicMock)  # Mock the open function for file saving
@patch.object(Path, 'mkdir')  # Mock Path.mkdir() instead of os.makedirs
def test_vsco_scrape_images(mock_mkdir, mock_open, mock_webdriver, caplog):
    # Mock the browser instance and return mocked elements (images)
    mock_browser = MagicMock()
    mock_webdriver.return_value = mock_browser

    # Simulate finding image elements on the VSCO page
    mock_image_element = MagicMock()
    mock_image_element.get_attribute.return_value = "mock_image_url"  # Simulate image URLs
    mock_browser.find_elements_by_tag_name.return_value = [mock_image_element, mock_image_element]
    
    # Use the scraper as a context manager and capture logs
    with caplog.at_level(logging.INFO):
        with VSCOScraper(email="valid_email", password="valid_password") as scraper:
            scraper.get_vsco_pics(username="validuser", save_path="valid/path", n=2)

        # Check the log contains "Sources Retrieved Successfully"
        assert "Sources Retrieved Successfully" in caplog.text

    # Ensure that the browser quit at the end
    mock_browser.quit.assert_called_once()

    # Check that the necessary directories were created using Path.mkdir()
    mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    mock_open.assert_called()  # Check that open was called to simulate saving files
