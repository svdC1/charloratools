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
    !WARNING!
    Use inside 'with' block,Example:
    with VSCOScraper(init args) as vscos:
      vscos.get_vsco_pics(get_vsco_pics args)
    Context Manager Type Class to save all images of a VSCO's profile gallery
    using Selenium,WebDriver Manager,opencv and urllib
    :param email: VSCO account login email - VSCO demands login to load all
    images in a profile's gallery
    :param password: VSCO account password
    :param headless: Wether to initialize selenium webdriver in headless mode
    or not,defaults to True
    :param incognito: Wether to initialize selenium webdriver in incognito
    mode or not,defaults to True
    :param add_arguments: Addition list of arguments to pass to ChromeOptions
    when initializing selenium webdriver,defaults to None
    :param webdriver_wait_timeout: How many seconds to wait before throwing a
    timeout error when browser can't find an element,defaults to 10
    :param webpage_wait_time: How many seconds to wait for a webpage to load
    using 'time.sleep()',defaults to 3
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
        s = f"""
        VSCOScraper Instance, logged in as {self.email}
        Headless:{self.headless}\nIncognito:{self.incognito}
        WebDriverArugments:{self.add_arguments}
        """
        return s

    def __repr__(self):
        e = f"email={self.email}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_timeout}"
        return f"VSCOScraper({e}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        self.driver = self.vsco_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements
        # in headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()

    def vsco_sign_in(self):
        """
        Method to login into a VSCO account in a headless selenium webdriver
        and return the logged-in driver
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
        Method to download VSCO profile gallery images
        :param username: VSCO account username to download gallery images from
        :param save_path: Path to save the images gallery
        :param n: Number of times to scroll the page to load more images to
        download,defaults to 10
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
    !WARNING!
    Use inside 'with' block,Example:
    with XScraper(init args) as xs:
      xs.get_x_pics(get_x_pics args)
    Context Manager Type Class to save all images of a X's profile media using
    Selenium,WebDriver Manager,opencv and urllib
    :param username: X account login username - X demands login to see a
    profile's media
    :param password: X account password
    :param headless: Wether to initialize selenium webdriver in headless mode
    or not,defaults to True
    :param incognito: Wether to initialize selenium webdriver in incognito
    mode or not,defaults to True
    :param add_arguments: Addition list of arguments to pass to ChromeOptions
    when initializing selenium webdriver,defaults to None
    :param webdriver_wait_timeout: How many seconds to wait before throwing a
    timeout error when browser can't find an element,defaults to 10
    :param webpage_wait_time: How many seconds to wait for a webpage to load
    using 'time.sleep()',defaults to 3
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
        s = f"""XScraper Instance, logged in as {self.username}
                Headless:{self.headless}
                Incognito:{self.incognito}
                WebDriverArugments:{self.add_arguments}"""
        return s

    def __repr__(self):
        u = f"username={self.username}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_timeout}"
        return f"XScraper({u}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        self.driver = self.x_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements in
        # headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()

    def x_sign_in(self):
        """
        Method to log into an X account using a Selenium Webdriver
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
        Method to download X profile media images
        :param username: X account username to download images from
        :param save_path: Path to save the images
        :param n: Number of times to scroll the page to load more images to
        download,defaults to 10
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
    !WARNING!
    This is still quite unstable - Haven't really figured out how Instagram
    updates the displayed images dynamically in
    the HTML, so even when logged in the amount of images it's able to find
    varies, although when testing for profiles
    with a lot of pictures I was able to scrape consistently ~50 pictures, so
    keep that in mind and any contribution
    is welcome.
    !WARNING!
    Use inside 'with' block,Example:
    with InstagrasmScraper(init args) as instas:
      instas.get_feed_pics(get_feed_pics args)
    Context Manager Type Class to save all images of a Instagram's profile
    media using Selenium,WebDriver Manager,opencv and urllib
    :param username: Instagram account login username
    :param password: Instagram account password
    :param headless: Wether to initialize selenium webdriver in headless mode
    or not,defaults to True
    :param incognito: Wether to initialize selenium webdriver in incognito
    mode or not,defaults to True
    :param add_arguments: Addition list of arguments to pass to ChromeOptions
    when initializing selenium webdriver,defaults to None
    :param webdriver_wait_timeout: How many seconds to wait before throwing a
    timeout error when browser can't find an element,defaults to 10
    :param webpage_wait_time: How many seconds to wait for a webpage to load
    using 'time.sleep()',defaults to 3
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
        s = """
        InstagramScraper Instance, logged in as {self.username}
        Headless:{self.headless}
        Incognito:{self.incognito}
        WebDriverArugments:{self.add_arguments}
        """
        return s

    def __repr__(self):
        u = f"username={self.username}"
        p = f"password={self.password}"
        h = f"headless={self.headless}"
        i = f"incognito={self.incognito}"
        ad = f"add_arguments={self.add_arguments}"
        wt = f"webdriver_timeout={self.webdriver_timeout}"
        return f"InstagramScraper({u}, {p}, {h}, {i}, {ad}, {wt})"

    def __enter__(self):
        self.driver = self.insta_sign_in()
        # -Need to set window size for ChromeWebdriver to find elements
        # in headless mode
        self.driver.set_window_size(1440, 900)
        self.logger.info('Finished Initialization Sucessfully')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()

    def insta_sign_in(self):
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
        Method to download Instagram profile feed images
        :param username: Instagram account username to download images from
        :param save_path: Path to save the images
        :param n: Number of times to scroll the page to load more images to
        download,defaults to 10
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
