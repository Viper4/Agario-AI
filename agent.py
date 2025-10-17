import numpy as np
import cv2
import pyautogui
import time
import threading
import json
from image_processing import ImageProcessing
from game import Game, Point
import random


class AgarAgent(threading.Thread):
    def __init__(self):
        super().__init__()
        self.program_running = True
        self.agent_running = False
        self.game = Game()

        with open("settings.json", encoding='utf-8') as file:
            self.settings = json.load(file)
        self.image_processor = ImageProcessing(self.settings["screen_res"], self.settings["image_scale"])

        print("AgarAI initialized")

    def run(self):
        random_id = random.randint(0, 10000)  # ID for this agent, can use as name in game to differentiate
        self.game.start_time = time.time()
        while self.program_running:
            if self.agent_running:
                img = self.image_processor.screenshot()
                objects = self.image_processor.object_recognition(img, False)
                if len(objects) > 0:
                    print(objects)
                self.image_processor.show_visual()
            time.sleep(0.1)
