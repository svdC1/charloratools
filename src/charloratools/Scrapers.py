"""
> Provides **context manager classes for scraping and
downloading media images from social media platforms.**

> Leverages **Selenium WebDriver for web automation to facilitate the
login, navigation, and retrieval of images from user profiles.**

Classes
-------
VSCOScraper
    Manages the scraping of images from a VSCO user's profile gallery.
    It handles user authentication and image downloading, ensuring resources
    are managed correctly.
XScraper
    Manages the scraping of media images from an X user's profile.
    Facilitates user authentication and downloading media.
InstagramScraper
    Manages the scraping of media images from an Instagram user's profile.
    This includes user authentication and downloading media.

Examples
--------
> Create an instance of the desired scraper class **within a context**
`(with statement)` to ensure proper resource management.

```python
from charloratools.Scrapers import VSCOScraper
    with VSCOScraper(*args) as scraper:
        scraper.get_vsco_pics()
```

Raises
------
VSCOSignInError
    Raised when there is an error during the login process to a VSCO account.
InstaSignInError
    Raised when there is an error during the login process to an
    Instagram account.
XSignInError
    Raised when there is an error during the login process to an X account.
NoImagesFoundError
    Raised when no images are found during the scraping process.
"""

# Default Python Libs imports
import logging
import time
# External Libs imports
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
# Scripts imports
from . import errors
from . import utils


class VSCOScraper:
    """
    Context Manager Type Class to save all images of a VSCO profile
    gallery using Selenium.
    """
    LOGIN_CSS_SELECTORS = {'user_input': '#identity',
                           'password': '#password',
                           'login_button': '#loginButton'}
    VSCO_BASE_URL = 'https://vsco.co/'
    VSCO_LOGIN_URL = 'https://vsco.co/user/login'
    LOAD_MORE_BUTTON = '#root > grain-theme > div.css-1n2jhjr.e1lxikmc0 > '
    LOAD_MORE_BUTTON += 'div.css-1fod2s4.e1lxikmc0 > '
    LOAD_MORE_BUTTON += 'main > div.css-87mlbn.e1c94cdo0 > '
    LOAD_MORE_BUTTON += 'div.css-1tkfhu2.e19mt3zn0 > grain-button'
    GALLERY_CSS_SELECTORS = {
        'load_more_button': LOAD_MORE_BUTTON,
        'footer_button': '#footer-about_vsco'}

    def __init__(self, email: str, password: str, headless: bool = True,
                 incognito: bool = True, add_arguments: list | None = None,
                 webdriver_wait_timeout: int = 10,
                 webpage_wait_time: int = 3):
        """
        Initializes the VSCOScraper with the specified login
        credentials and options.

        Parameters
        ----------
        email : str
            The VSCO account login email.
        password : str
            The VSCO account password.
        headless : bool, optional
            Whether to run the Selenium WebDriver in headless mode.
            Defaults to True.
        incognito : bool, optional
            Whether to run the Selenium WebDriver in incognito mode.
            Defaults to True.
        add_arguments : list or None, optional
            Additional arguments for ChromeOptions when initializing the
            WebDriver. Defaults to None.
        webdriver_wait_timeout : int, optional
            Number of seconds to wait before a timeout error is raised when
            the browser cannot find an element. Defaults to 10.
        webpage_wait_time : int, optional
            Number of seconds to wait for a webpage to load using
            'time.sleep()'. Defaults to 3.

        Raises
        ------
        VSCOSignInError
            If an error occurs during the login process.
        """
        self.logger = logging.getLogger('VSCO-SCRAPER')
        self.email = email
        self.password = password
        self.headless = headless
        self.incognito = incognito
        self.add_arguments = add_arguments
        self.webdriver_timeout = webdriver_wait_timeout
        self.webpage_wait_time = webpage_wait_time
        self.driver = None

    def __str__(self):
        """
        Returns a string representation of the VSCOScraper instance.
        This representation includes the login email and configuration
        details like headless mode status, incognito status,
        and additional WebDriver arguments.

        Returns
        -------
        str
            A formatted string summarizing the state of the VSCOScraper
            instance.
        """
        s = f"""
        VSCOScraper Instance, logged in as {self.email}
        Headless:{self.headless}\nIncognito:{self.incognito}
        WebDriverArugments:{self.add_arguments}
        """
        return s

    def __repr__(self):
        """
        Returns a detailed string representation of the VSCOScraper instance.

        This representation provides a concise view of the instance variables
        including email and headless/incognito mode settings.

        Returns
        -------
        str
            A formatted string for the VSCOScraper instance,
            suitable for debugging.
        """
        e = f"email={self.email}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_timeout}"
        return f"VSCOScraper({e}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        """
        Initializes the WebDriver and performs login before yielding the
        scraper instance.
        Returns
        -------
        VSCOScraper
            The instance of VSCOScraper for use within the 'with' block.
        """
        self.driver = self.vsco_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements
        # in headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleans up the WebDriver when exiting the context.
        This method ensures that the WebDriver quits, freeing resources.

        Parameters
        ----------
        exc_type : type or None
            The exception class raised, if any.
        exc_val : Exception or None
            The exception instance raised, if any.
        exc_tb : traceback or None
            The traceback object, if any.
        """
        if self.driver:
            self.driver.quit()

    def vsco_sign_in(self):
        """
        Logs into a VSCO account using the Selenium WebDriver.

        This method performs the login action by interacting with the login
        elements of the VSCO website.

        Returns
        -------
        webdriver
            The Selenium WebDriver instance after logging in successfully.

        Raises
        ------
        VSCOSignInError
            If an error occurs during the login process.
        """
        driver = utils.initialize_driver(
            headless=self.headless, incognito=self.incognito,
            add_arguments=self.add_arguments)
        self.logger.info('Webdriver Initialized Sucessfully')
        try:
            wait = WebDriverWait(driver, self.webdriver_timeout)
            # Go to the website's main page
            driver.get(VSCOScraper.VSCO_LOGIN_URL)
            # Wait for the page to load
            time.sleep(self.webpage_wait_time)
            self.logger.info("Entering user email")
            user_input = wait.until(EC.visibility_of_element_located((
                By.CSS_SELECTOR, VSCOScraper.LOGIN_CSS_SELECTORS['user_input']
                )))
            user_input.send_keys(self.email)
            self.logger.info("Entering password")
            password_input = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR, VSCOScraper.LOGIN_CSS_SELECTORS['password'])
                ))
            password_input.send_keys(self.password)
            time.sleep(self.webpage_wait_time)
            self.logger.info("Clicking final sign in button")
            final_sign_in_button = wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR,
                 VSCOScraper.LOGIN_CSS_SELECTORS['login_button']
                 )))
            final_sign_in_button.click()
            time.sleep(self.webpage_wait_time)
            self.logger.info("Signed in successfully")
            self.wait = wait
            return driver
        except Exception as e:
            driver.quit()
            raise errors.VSCOSignInError(str(e))

    def get_vsco_pics(self, username: str, save_path: str, n: int = 10):
        """
        Downloads images from a specified VSCO profile gallery.

        This method retrieves images from the specified user's gallery by
        scrolling the page and extracting image source URLs.

        Parameters
        ----------
        username : str
            The username of the VSCO account to scrape images from.
        save_path : str
            The path where the downloaded images will be saved.
        n : int, optional
            The number of times to scroll down the page to load more images.
            Defaults to 10.

        Raises
        ------
        NoImagesFoundError
            If no images are found during the scraping process.
        Exception
            If any other error occurs during the image retrieval process.
        """
        # Checking save_dir
        save_path = utils.dirisvalid(save_path, create_if_not_found=True)
        user_url = VSCOScraper.VSCO_BASE_URL + username
        try:
            self.driver.get(user_url)
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(self.webpage_wait_time)
            try:
                load_more_button = self.wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR,
                     VSCOScraper.GALLERY_CSS_SELECTORS['load_more_button']
                     )))
                time.sleep(self.webpage_wait_time)
                load_more_button.click()
                utils.page_scroll(self.driver, n, self.webpage_wait_time)
            except TimeoutException:
                tw = """Timed Out While Looking for load more button,saving
                        images currently displayed"""
                self.logger.warning(tw)

            imgs = self.driver.find_elements(By.TAG_NAME, 'img')
            srcs = []

            if imgs:
                self.logger.info(f'Found {len(imgs)} images')
                for i, img in zip(range(len(imgs)), imgs):
                    src = img.get_attribute('src')
                    srcs.append(src)
                self.logger.info("Sources Retrieved Sucessfully")

            else:
                raise errors.NoImagesFoundError('No Images Found')

            utils.download_from_src(
                srcs=srcs, prefix=username, save_path=save_path,
                logger=self.logger)
        except Exception as e:
            raise Exception(str(e))


class XScraper:
    """
    Context manager for scraping images from an X user's profile media.
    """
    U = ['#layers > div:nth-child(2) > div > div > div > div > div > ',
         'div.css-175oi2r.r-1ny4l3l.r-18u37iz.r-1pi2tsx.r-1777fci.r-1xcajam.r',
         '-ipm5af.r-g6jmlv.r-1awozwy > div.css-175oi2r.r-1wbh5a2.r-htvplk.r-',
         '1udh08x.r-1867qdf.r-kwpbio.r-rsyp9y.r-1pjcn9w.r-1279nm1 > ',
         'div > div > div.css-175oi2r.r-1ny4l3l.r-6koalj.r-16y2uox.r-kemksi.r',
         '-1wbh5a2 > div.css-175oi2r.r-16y2uox.r-1wbh5a2.r-f8sm7e.r-13qz1uu.r',
         '-1ye8kvj > div > div > div > div.css-175oi2r.r-1mmae3n.r-1e084wi.r-',
         '13qz1uu > label > div > div.css-175oi2r.r-18u37iz.r-16y2uox.r-1',
         'wbh5a2.r-1wzrnnt.r-1udh08x.r-xd6kpl.r-is05cd.r-ttdzmv > div > input']
    U = ''.join(U)
    NB = ['#layers > div:nth-child(2) > div > div > div > div > div > ',
          'div.css-175oi2r.r-1ny4l3l.r-18u37iz.r-1pi2tsx.r-1777fci.r-1xcajam.',
          'r-ipm5af.r-g6jmlv.r-1awozwy > div.css-175oi2r.r-1wbh5a2.r-htvplk.',
          'r-1udh08x.r-1867qdf.r-kwpbio.r-rsyp9y.r-1pjcn9w.r-1279nm1 > div >',
          'div > div.css-175oi2r.r-1ny4l3l.r-6koalj.r-16y2uox.r-kemksi.',
          'r-1wbh5a2 > div.css-175oi2r.r-16y2uox.r-1wbh5a2.r-f8sm7e.r-13qz1uu',
          '.r-1ye8kvj > div > div > div > button:nth-child(6)']
    NB = ''.join(NB)
    P = ['#layers > div:nth-child(2) > div > div > div > div > div >',
         'div.css-175oi2r.r-1ny4l3l.r-18u37iz.r-1pi2tsx.r-1777fci.r-1xcajam.',
         'r-ipm5af.r-g6jmlv.r-1awozwy > div.css-175oi2r.r-1wbh5a2.r-htvplk.',
         'r-1udh08x.r-1867qdf.r-kwpbio.r-rsyp9y.r-1pjcn9w.r-1279nm1 > div ',
         '> div > div.css-175oi2r.r-1ny4l3l.r-6koalj.r-16y2uox.r-kemksi.',
         'r-1wbh5a2 > div.css-175oi2r.r-16y2uox.r-1wbh5a2.r-f8sm7e.r-13qz1uu.',
         'r-1ye8kvj > div.css-175oi2r.r-16y2uox.r-1wbh5a2.r-1dqxon3 > div ',
         '> div > div.css-175oi2r.r-1e084wi.r-13qz1uu > div > label > div ',
         '> div.css-175oi2r.r-18u37iz.r-16y2uox.r-1wbh5a2.r-1wzrnnt.r-1udh08x',
         '.r-xd6kpl.r-is05cd.r-ttdzmv > div.css-146c3p1.r-bcqeeo.r-1ttztb7.',
         'r-qvutc0.r-37j5jr.r-135wba7.r-16dba41.r-1awozwy.r-6koalj.',
         'r-1inkyih.r-13qz1uu > input']
    P = ''.join(P)
    SB = ['#layers > div:nth-child(2) > div > div > div > div > div > ',
          'div.css-175oi2r.r-1ny4l3l.r-18u37iz.r-1pi2tsx.r-1777fci.',
          'r-1xcajam.r-ipm5af.r-g6jmlv.r-1awozwy > div.css-175oi2r.',
          'r-1wbh5a2.r-htvplk.r-1udh08x.r-1867qdf.r-kwpbio.r-rsyp9y',
          '.r-1pjcn9w.r-1279nm1 > div > div > div.css-175oi2r.r-1ny4l3l',
          '.r-6koalj.r-16y2uox.r-kemksi.r-1wbh5a2 > div.css-175oi2r.r-16y2uox',
          '.r-1wbh5a2.r-f8sm7e.r-13qz1uu.r-1ye8kvj > div.css-175oi2r.r-1f0wa7',
          'y > div > div.css-175oi2r > div > div > button']
    SB = ''.join(SB)
    XLOGIN_CSS_SELECTORS = {'username': U,
                            'next_button': NB,
                            'password': P,
                            'sign_in_button': SB
                            }
    XBASE_URL = "https://www.x.com/"
    XLOGIN_URL = "https://www.x.com/login"

    def __init__(self, username: str, password: str, headless: bool = True,
                 incognito: bool = True, add_arguments: list | None = None,
                 webdriver_wait_timeout: int = 10, webpage_wait_time: int = 3):
        """
        Initializes the XScraper with the specified login
        credentials and options.

        Parameters
        ----------
        username : str
            The X account login username.
        password : str
            The X account password.
        headless : bool, optional
            Whether to run the Selenium WebDriver in headless mode.
            Defaults to True.
        incognito : bool, optional
            Whether to run the WebDriver in incognito mode. Defaults to True.
        add_arguments : list or None, optional
            Additional arguments for ChromeOptions when initializing the
            WebDriver. Defaults to None.
        webdriver_wait_timeout : int, optional
            Number of seconds to wait before a timeout error is raised when
            the browser cannot find an element. Defaults to 10.
        webpage_wait_time : int, optional
            Number of seconds to wait for a webpage to load using
            'time.sleep()'. Defaults to 3.

        Raises
        ------
        XSignInError
            If there is an error during the login process to the X account.
        """
        self.logger = logging.getLogger('X-SCRAPER')
        self.username = username
        self.password = password
        self.headless = headless
        self.incognito = incognito
        self.add_arguments = add_arguments
        self.webdriver_wait_timeout = webdriver_wait_timeout
        self.webpage_wait_time = webpage_wait_time
        self.driver = None

    def __str__(self):
        """
        Returns a string representation of the XScraper instance.

        This representation includes the login username and
        configuration details like headless mode status, incognito status,
        and additional WebDriver arguments.

        Returns
        -------
        str
            A formatted string summarizing the state of the XScraper instance.
        """
        s = f"""XScraper Instance, logged in as {self.username}
                Headless:{self.headless}
                Incognito:{self.incognito}
                WebDriverArugments:{self.add_arguments}"""
        return s

    def __repr__(self):
        """
        Returns a detailed string representation of the XScraper instance.

        This representation provides a concise view of the instance variables
        including username and headless/incognito mode settings.

        Returns
        -------
        str
            A formatted string for the XScraper instance,
            suitable for debugging.
        """
        u = f"username={self.username}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_wait_timeout}"
        return f"XScraper({u}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        """
        Initializes the WebDriver and performs login before yielding the
        scraper instance.

        Returns
        -------
        XScraper
            The instance of XScraper for use within the 'with' block.
        """
        self.driver = self.x_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements in
        # headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleans up by quitting the WebDriver when exiting the context.

        Parameters
        ----------
        exc_type : type or None
            The exception class raised, if any.
        exc_val : Exception or None
            The exception instance raised, if any.
        exc_tb : traceback or None
            The traceback object, if any.
        """
        if self.driver:
            self.driver.quit()

    def x_sign_in(self):
        """
        Logs into an X account using the Selenium WebDriver.

        This method performs the login action by interacting with the login
        elements of the X website.

        Returns
        -------
        webdriver
            The Selenium WebDriver instance after logging in successfully.

        Raises
        ------
        XSignInError
            If an error occurs during the login process.
        """
        driver = utils.initialize_driver(
            headless=self.headless, incognito=self.incognito,
            add_arguments=self.add_arguments)
        self.logger.info('Webdriver Initialized Sucessfully')
        try:
            wait = WebDriverWait(driver, self.webdriver_wait_timeout)
            driver.get(XScraper.XLOGIN_URL)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Entering Username')
            username_input = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR, XScraper.XLOGIN_CSS_SELECTORS['username'])))
            username_input.send_keys(self.username)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Clicking Next Button')
            next_button = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR,
                 XScraper.XLOGIN_CSS_SELECTORS['next_button'])))
            next_button.click()
            time.sleep(self.webpage_wait_time)
            self.logger.info('Entering Password')
            password_input = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR, XScraper.XLOGIN_CSS_SELECTORS['password'])))
            password_input.send_keys(self.password)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Clicking SignIn Button')
            signin_button = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR,
                 XScraper.XLOGIN_CSS_SELECTORS['sign_in_button'])))
            signin_button.click()
            time.sleep(self.webpage_wait_time)
            self.logger.info('Logged in Sucessfully')
            self.wait = wait
            return driver
        except Exception as e:
            driver.quit()
            raise errors.XSignInError(f'Error Loggging in to X : {e}')

    def get_x_pics(self, username: str, save_path: str, n: int = 10):
        """
        Downloads media images from a specified X profile.

        This method retrieves images from the specified user's profile by
        scrolling the page and extracting image source URLs.

        Parameters
        ----------
        username : str
            The username of the X account to scrape media images from.
        save_path : str
            The path where the downloaded images will be saved.
        n : int, optional
            The number of times to scroll down the page to load more images.
            Defaults to 10.

        Raises
        ------
        NoImagesFoundError
            If no images are found during the scraping process.
        Exception
            If any error occurs during the image retrieval process.
        """
        save_path = utils.dirisvalid(save_path, create_if_not_found=True)
        try:
            self.driver.get(f'https://x.com/{username}/media')
            utils.page_scroll(self.driver, n, self.webpage_wait_time)
            imgs = self.driver.find_elements(By.TAG_NAME, 'img')
            srcs = []
            if imgs:
                for i, img in zip(range(len(imgs)), imgs):
                    src = img.get_attribute('src')
                    srcs.append(src)
                self.logger.info("Sources Retrieved Sucessfully")
            else:
                raise errors.NoImagesFoundError('No Images Found')
            fmt_srcs = [src.split('&')[0] for src in srcs if '/media/' in src]
            self.logger.info(f'Found {len(fmt_srcs)} images')
            utils.download_from_src(fmt_srcs, username, save_path, self.logger)
        except Exception as e:
            raise Exception(str(e))


class InstagramScraper:
    """
    Context manager for scraping images from an Instagram user's profile media.
    """
    U = '#loginForm > div > div:nth-child(1) > div > label > input'
    P = '#loginForm > div > div:nth-child(2) > div > label > input'
    SB = '#loginForm > div > div:nth-child(3) > button'
    INSTA_LOGIN_CSS_SELECTORS = {'username': U,
                                 'password': P,
                                 'sign_in_button': SB
                                 }
    INSTA_BASE_URL = "https://www.instagram.com/"

    ImgD = ['#mount_0_0_lz > div > div > div.x9f619.x1n2onr6.x1ja2u2z > div ',
            '> div > div.x78zum5.xdt5ytf.x1t2pt76.x1n2onr6.x1ja2u2z.x10cihs4 ',
            '> div:nth-child(2) > div > div.x1gryazu.xh8yej3.x10o80wk.x14k21',
            'rp.x17snn68.x6osk4m.x1porb0y.x8vgawa > section > main > div',
            '> div:nth-child(3)']
    INSTA_GALLERY_CSS_SELECTORS = {
        'imgs_display': ImgD}

    def __init__(self, username: str, password: str, headless: bool = True,
                 incognito: bool = True, add_arguments: list | None = None,
                 webdriver_wait_timeout: int = 10,
                 webpage_wait_time: int = 3):
        """
        Initializes the InstagramScraper with login credentials and options.

        Parameters
        ----------
        username : str
            The Instagram account login username.
        password : str
            The Instagram account password.
        headless : bool, optional
            Whether to run the Selenium WebDriver in headless mode.
            Defaults to True.
        incognito : bool, optional
            Whether to run the WebDriver in incognito mode. Defaults to True.
        add_arguments : list or None, optional
            Additional arguments for ChromeOptions when initializing the
            WebDriver. Defaults to None.
        webdriver_wait_timeout : int, optional
            Number of seconds to wait before a timeout error is raised when an
            element can't be found. Defaults to 10.
        webpage_wait_time : int, optional
            Number of seconds to wait for a webpage to load using
            'time.sleep()'. Defaults to 3.

        Raises
        ------
        InstaSignInError
            If there is an error during the login process to the
            Instagram account.
        """
        self.logger = logging.getLogger('INSTA-SCRAPER')
        self.username = username
        self.password = password
        self.headless = headless
        self.incognito = incognito
        self.add_arguments = add_arguments
        self.webdriver_wait_timeout = webdriver_wait_timeout
        self.webpage_wait_time = webpage_wait_time
        self.driver = None

    def __str__(self):
        """
        Returns a string representation of the InstagramScraper instance.

        This representation includes the login username and
        configuration details like headless mode status, incognito status,
        and additional WebDriver arguments.

        Returns
        -------
        str
            A formatted string summarizing the state of the
            InstagramScraper instance.
        """

        s = """
        InstagramScraper Instance, logged in as {self.username}
        Headless:{self.headless}
        Incognito:{self.incognito}
        WebDriverArugments:{self.add_arguments}
        """
        return s

    def __repr__(self):
        """
        Returns a detailed string representation of the
        InstagramScraper instance.

        This representation provides a concise view of the instance variables
        including username and headless/incognito mode settings.

        Returns
        -------
        str
            A formatted string for the InstagramScraper instance,
            suitable for debugging.
        """
        u = f"username={self.username}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_timeout}"
        return f"InstagramScraper({u}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        """
        Initializes the WebDriver and performs login before yielding
        the scraper instance.

        Returns
        -------
        InstagramScraper
            The instance of InstagramScraper for use within the 'with' block.
        """
        self.driver = self.insta_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements
        # in headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Cleans up the WebDriver when exiting the context.

        This method ensures that the WebDriver quits, freeing resources.

        Parameters
        ----------
        exc_type : type or None
            The exception class raised, if any.
        exc_val : Exception or None
            The exception instance raised, if any.
        exc_tb : traceback or None
            The traceback object, if any.
        """
        if self.driver:
            self.driver.quit()

    def insta_sign_in(self):
        """
        Logs into an Instagram account using the Selenium WebDriver.

        This method performs the login action by interacting with the login
        elements of the Instagram website.

        Returns
        -------
        webdriver
            The Selenium WebDriver instance after successfully logging in.

        Raises
        ------
        InstaSignInError
            If an error occurs during the login process.
        """
        driver = utils.initialize_driver(
            headless=self.headless, incognito=self.incognito,
            add_arguments=self.add_arguments)
        self.logger.info('Webdriver Initialized Sucessfully')
        try:
            wait = WebDriverWait(driver, self.webdriver_wait_timeout)
            driver.get(InstagramScraper.INSTA_BASE_URL)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Entering Username')
            username_input = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR,
                 InstagramScraper.INSTA_LOGIN_CSS_SELECTORS['username'])))
            username_input.send_keys(self.username)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Entering Password')
            password_input = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR,
                 InstagramScraper.INSTA_LOGIN_CSS_SELECTORS['password'])))
            password_input.send_keys(self.password)
            time.sleep(self.webpage_wait_time)
            self.logger.info('Clicking SignIn Button')
            signin_button = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR,
                 InstagramScraper.INSTA_LOGIN_CSS_SELECTORS['sign_in_button'])
                ))
            signin_button.click()
            time.sleep(self.webpage_wait_time)
            self.logger.info('Logged in Sucessfully')
            self.wait = wait
            return driver
        except Exception as e:
            driver.quit()
            raise errors.InstaSignInError(f'Error Loggging in : {e}')

    def get_feed_pics(self, username: str, save_path: str, n: int = 10):
        """
        Downloads feed images from a specified Instagram profile.

        This method retrieves images from the specified user's feed by
        scrolling the page and extracting image source URLs.

        Parameters
        ----------
        username : str
            The username of the Instagram account to scrape images from.
        save_path : str
            The path where the downloaded images will be saved.
        n : int, optional
            The number of times to scroll down the page to load more images.
            Defaults to 10.

        Raises
        ------
        NoImagesFoundError
            If no images are found during the scraping process.
        Exception
            If any error occurs during the image retrieval process.
        """
        save_path = utils.dirisvalid(save_path, create_if_not_found=True)
        try:
            self.driver.get(f'https://instagram.com/{username}')
            utils.page_scroll(self.driver, n, self.webpage_wait_time)
            time.sleep(self.webpage_wait_time)
            imgs = self.driver.find_elements(By.TAG_NAME, 'img')
            srcs = []
            if imgs:
                for i, img in zip(range(len(imgs)), imgs):
                    src = img.get_attribute('src')
                    srcs.append(src)
                self.logger.info("Sources Retrieved Sucessfully")
            else:
                raise errors.NoImagesFoundError('No Images Found')
            time.sleep(self.webpage_wait_time)
            utils.download_from_src(srcs, username, save_path, self.logger)
        except Exception as e:
            raise Exception(str(e))
