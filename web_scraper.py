import random
import os
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By, ByType
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys


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
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        # User agents to rotate through
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

        options.add_argument(f"user-agent={random.choice(user_agents)}")  # Pick random user agent to use

        options.add_argument("--host-rules=MAP ads.google.com 127.0.0.1,MAP doubleclick.net 127.0.0.1,MAP adservice.google.com 127.0.0.1")

        service = Service(executable_path=os.path.join(PROJECT_DIR, "WebStuff", "msedgedriver.exe"))  # Define custom webdriver executable
        # driver = webdriver.Chrome(options, service)
        self.driver = webdriver.Edge(options, service)

        self.driver.get("https://agar.io/#ffa")
        self.main_ui = self.driver.find_element("id", "mainui-app")
        self.canvas = self.driver.find_element("tag name", "canvas")
        self.actions = ActionChains(self.driver)

        # Run JavaScript in console to remove ads periodically
        hide_ads_script = """
        setInterval(() => {
            ['iframe', '#adsBottom', '#adbg', '#openfl-content'].forEach(s => {
                document.querySelectorAll(s).forEach(e => e.remove());
            });
        }, 500);
        """
        self.driver.execute_script(hide_ads_script)

        self.canvas_png = None

    def screenshot_canvas_image(self):
        """
        Takes screenshot of the game canvas and returns it as bytes.
        :return: bytes
        """
        self.canvas_png = self.canvas.screenshot_as_png
        return self.canvas_png

    def enter_name(self, name: str, wait: bool):
        """
        Enters a name into the input field (if it exists) for the name when you start the game.
        Returns True on success, False otherwise.
        :param name: str
        :param wait: If we should wait for the name input
        :return: bool
        """
        try:
            if wait:
                if not self.wait_for_element(By.ID, "nick", 5):
                    return False
            name_input = self.driver.find_element(By.ID, "nick")
            if not name_input.is_displayed():
                return False
            name_input.clear()
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

    def play_game(self, wait: bool):
        """
        Presses the play button if it exists.
        Returns True on success, False otherwise.
        :return: bool
        """
        try:
            if wait:
                if not self.wait_for_element(By.ID, "play", 5):
                    return False
            play_button = self.driver.find_element(By.ID, "play")
            # Check if the button is hidden
            if not play_button.is_displayed():
                return False
            play_button.click()
            return True
        except NoSuchElementException or ElementNotInteractableException:
            return False

    def press_continue(self, wait: bool):
        """
        Presses the continue button if it exists.
        Returns True on success, False otherwise.
        :param: If we should wait for the continue button
        :return: bool
        """
        try:
            if wait:
                if not self.wait_for_element("id", "statsContinue", 5):
                    return False
            continue_button = self.driver.find_element("id", "statsContinue")
            if not continue_button.is_displayed():
                return False
            continue_button.click()
            return True
        except NoSuchElementException or ElementNotInteractableException:
            return False

    def wait_for_element(self, by: ByType, value: str, timeout: float):
        """
        Waits for an element to be present and displayed on the page.
        Returns True on success, False otherwise.
        :param by: str
        :param value: str
        :param timeout: float
        :return: bool
        """
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((by, value)))
            WebDriverWait(self.driver, timeout).until(EC.visibility_of_element_located((by, value)))
        except TimeoutException:
            return False
        return True

    def get_stats(self, wait: bool):
        """
        Find stats once game ends (we die or time runs out)
        :return: (food_eaten, time_alive, cells_eaten, highest_mass)
        """
        try:
            if wait:
                if not self.wait_for_element(By.CLASS_NAME, "stats-food-eaten", 5):
                    return None
            food_element = self.driver.find_element(By.CLASS_NAME, "stats-food-eaten")
            time_alive_element = self.driver.find_element(By.CLASS_NAME, "stats-time-alive")
            cells_element = self.driver.find_element(By.CLASS_NAME, "stats-cells-eaten")
            highest_mass_element = self.driver.find_element(By.CLASS_NAME, "stats-highest-mass")

            split_time = time_alive_element.text.split(":")
            while len(split_time) < 3:
                split_time.insert(0, "0")
            hours, minutes, seconds = split_time
            time_alive = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

            return int(food_element.text), time_alive, int(cells_element.text), int(highest_mass_element.text)
        except NoSuchElementException or ElementNotInteractableException:
            return None

    def press_space(self):
        """
        Sends space key to split
        """
        self.actions.send_keys(Keys.SPACE).perform()

    def press_w(self):
        """
        Sends w key to eject
        """
        self.actions.send_keys("w").perform()

    def move(self, x, y, scale):
        """
        Moves the mouse to the center of the screen with offset (x, y)*scale.
        The cursor placement needed to move can somtimes vary depending on how the player split,
        but offsetting from the center of the screen is a good enough approximation.
        :param x: float
        :param y: float
        :param scale: float
        """
        self.actions.move_to_element_with_offset(self.canvas, x, y).perform()
