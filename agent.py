import time
import threading
import json
from image_processing import ImageProcessing
from web_scraper import WebScraper
from game import Game, Point
import random


class AgarAgent(threading.Thread):
    def __init__(self, run_interval: float):
        super().__init__()
        self.run_interval = run_interval
        self.program_running = True
        self.agent_running = False
        self.game = Game()

        with open("settings.json", encoding='utf-8') as file:
            self.settings = json.load(file)
        self.scraper = WebScraper()
        self.image_processor = ImageProcessing()

        print("AgarAI initialized")

    def run(self):
        random_id = random.randint(0, 10000)  # ID for this agent, can use as name in game to differentiate it (maybe)

        self.scraper.wait_for_element("id", "nick", 10)  # Wait until name input field is loaded
        time.sleep(1)  #
        if not self.scraper.enter_name(random_id):
            print("Failed to enter name")
            return

        self.scraper.wait_for_element("id", "play", 10)  # Wait until play button is loaded
        if not self.scraper.play_game():
            print("Failed to play game")
            return

        self.game.start_time = time.time()
        while self.program_running:
            if self.agent_running:
                if self.scraper.in_game():
                    canvas_png = self.scraper.get_canvas_image()
                    img = self.image_processor.convert_to_mat(canvas_png)
                    objects = self.image_processor.object_recognition(img, False)
                    self.image_processor.show_visual()
            time.sleep(self.run_interval)
