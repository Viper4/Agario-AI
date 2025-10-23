import random
import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory of project


class WebScraper:
    def __init__(self):
        options = Options()

        # Set up web driver options to avoid bot detection
        options.add_experimental_option("detach", True)  # Keep browser open

        # Arguments to prevent bot detection
        options.add_argument("--no-sandbox")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("disable-infobars")

        # Scrape list of user agents to use
        user_agents = ["Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.10 Safari/605.1.1",
                       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.3",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3",
                       "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.3",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Trailer/93.3.8652.5",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 OPR/117.0.0.",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.1958",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.3"]
        #response = requests.get("https://www.useragents.me/api").json()  # API endpoint doesn't work anymore
        #for entry in response["data"]:
        #    user_agents.append(entry["ua"])

        options.add_argument(f"user-agent={random.choice(user_agents)}")  # Pick random user agent to use

        service = Service(executable_path=PROJECT_DIR + "\\msedgedriver.exe")  # Define custom webdriver executable
        # driver = webdriver.Chrome(options, service)
        self.driver = webdriver.Edge(options, service)

        self.driver.get("https://agar.io/#ffa")

        self.main_ui = self.driver.find_element("id", "mainui-app")

    def get_canvas_image(self):
        """
        Takes screenshot of the game canvas and returns it as bytes.
        :return: bytes
        """
        canvas = self.driver.find_element("tag name", "canvas")
        return canvas.screenshot_as_png

    def enter_name(self, name):
        """
        Enters a name into the input field (if it exists) for the name when you start the game.
        Returns True on success, False otherwise.
        :param name: string
        :return: bool
        """
        try:
            name_input = self.driver.find_element("id", "nick")
            if not name_input.is_displayed():
                return False
            name_input.send_keys(name)
            return True
        except NoSuchElementException or ElementNotInteractableException:
            return False

    def in_game(self):
        """
        Check whether we are actively playing the game or in the menu/watching ad.
        :return: bool
        """
        # main_ui element is disabled from display when we're playing
        return not self.main_ui.is_displayed()

    def play_game(self):
        """
        Presses the play button if it exists.
        Returns True on success, False otherwise.
        :return: bool
        """
        try:
            play_button = self.driver.find_element("id", "play")
            # Check if the button is hidden
            if not play_button.is_displayed():
                return False
            play_button.click()
            return True
        except NoSuchElementException or ElementNotInteractableException:
            return False

    def press_continue(self):
        """
        Presses the continue button if it exists.
        Returns True on success, False otherwise.
        :return: bool
        """
        try:
            continue_button = self.driver.find_element("id", "statsContinue")
            if not continue_button.is_displayed():
                return False
            continue_button.click()
            return True
        except NoSuchElementException or ElementNotInteractableException:
            return False

    def wait_for_element(self, by: str, value: str, timeout: float):
        """
        Waits for an element to be present and displayed on the page.
        Returns True on success, False otherwise.
        :param by: string
        :param value: string
        :param timeout: float
        :return: bool
        """
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((by, value)))
            WebDriverWait(self.driver, timeout).until(EC.visibility_of_element_located((by, value)))
        except TimeoutException:
            return False
        return True
