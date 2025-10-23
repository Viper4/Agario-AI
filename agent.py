import numpy as np
import cv2
import pyautogui
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
        self.game.start_time = time.time()
        while self.program_running:
            if self.agent_running:
                canvas_png = self.scraper.get_canvas_image()
                img = self.image_processor.convert_to_mat(canvas_png)
                objects = self.image_processor.object_recognition(img, False)
                self.image_processor.show_visual()
            time.sleep(self.run_interval)
