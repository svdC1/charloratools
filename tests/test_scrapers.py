from unittest.mock import patch, MagicMock
from charloratools.Scrapers import VSCOScraper, XScraper
from charloratools.Scrapers import InstagramScraper
from charloratools.errors import VSCOSignInError, NoImagesFoundError
from charloratools.errors import XSignInError, InstaSignInError
import pytest


def test_vsco_scraper_success(mock_webdriver):
    mock_webdriver.get.return_value = None
    mock_webdriver.find_element_by_name.return_value = MagicMock()
    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.VSCOScraper.get_vsco_pics',
                   return_value=['image1.jpg', 'image2.jpg']):
            with VSCOScraper('username', 'password') as scraper:
                images = scraper.get_vsco_pics('test_username',
                                               'save_path')
                assert images == ['image1.jpg', 'image2.jpg']


def test_vsco_scraper_login_failure(mock_webdriver):
    mock_webdriver.find_element_by_name.side_effect = Exception("Login failed")
    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               side_effect=Exception("Login failed")):
        with pytest.raises(VSCOSignInError):
            with VSCOScraper('wrong_user', 'wrong_pass') as scraper:
                scraper.get_vsco_pics('test_username', 'save_path')


def test_vsco_no_images_found(mock_webdriver):
    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.VSCOScraper.get_vsco_pics',
                   side_effect=NoImagesFoundError):
            with pytest.raises(NoImagesFoundError):
                with VSCOScraper('username', 'password') as scraper:
                    scraper.get_vsco_pics('test_username', 'save_path')


def test_xscraper_success(mock_webdriver):
    mock_webdriver.get.return_value = None

    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.XScraper.get_x_pics',
                   return_value=['image1.jpg', 'image2.jpg']):
            with XScraper('username', 'password') as scraper:
                images = scraper.get_x_pics('test_username', 'save_path')
                assert images == ['image1.jpg', 'image2.jpg']


def test_xscraper_login_failure(mock_webdriver):
    mock_webdriver.find_element_by_name.side_effect = Exception("Login failed")

    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               side_effect=Exception("Login failed")):
        with pytest.raises(XSignInError):
            with XScraper('wrong_user', 'wrong_pass') as scraper:
                scraper.get_x_pics('test_username', 'save_path')


def test_xscraper_no_images_found(mock_webdriver):
    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.XScraper.get_x_pics',
                   side_effect=NoImagesFoundError):
            with pytest.raises(NoImagesFoundError):
                with XScraper('username', 'password') as scraper:
                    scraper.get_x_pics('test_username', 'save_path')


def test_instagram_scraper_success(mock_webdriver):
    mock_webdriver.get.return_value = None

    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.InstagramScraper.get_feed_pics',
                   return_value=['image1.jpg', 'image2.jpg']):
            with InstagramScraper('username', 'password') as scraper:
                images = scraper.get_feed_pics('test_username', 'save_path')
                assert images == ['image1.jpg', 'image2.jpg']


def test_instagram_scraper_login_failure(mock_webdriver):
    mock_webdriver.find_element_by_name.side_effect = Exception("Login failed")

    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               side_effect=Exception("Login failed")):
        with pytest.raises(InstaSignInError):
            with InstagramScraper('wrong_user', 'wrong_pass') as scraper:
                scraper.get_feed_pics('test_username', 'save_path')


def test_instagram_scraper_no_images_found(mock_webdriver):
    with patch('selenium.webdriver.support.ui.WebDriverWait.until',
               return_value=MagicMock()):
        with patch('charloratools.Scrapers.InstagramScraper.get_feed_pics',
                   side_effect=NoImagesFoundError):
            with pytest.raises(NoImagesFoundError):
                with InstagramScraper('username', 'password') as scraper:
                    scraper.get_feed_pics('test_username', 'save_path')
