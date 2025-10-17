import cv2
import numpy as np
import json
import pyautogui
import ctypes
from game import Object


class ImageProcessing:
    def __init__(self, screen_resolution, image_scale: float):
        self.masks = {}
        with open("masks.json", encoding='utf-8') as file:
            self.masks = json.load(file)

        self.screen_res = screen_resolution
        self.image_scale = image_scale
        if screen_resolution == "auto":
            self.screen_res = self.get_screen_res()
            print(f"Using screen resolution of {self.screen_res}")

        self.scaled_resolution = [int(self.screen_res[0] * self.image_scale),
                                  int(self.screen_res[1] * self.image_scale)]
        self.screen_offset = []

        self.img_visualization = None  # Image for visualizing processing results

    def get_screen_res(self):
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def screenshot(self):
        """
        Takes a screenshot
        :return: cv2.Mat
        """
        self.screen_offset = [self.screen_res[0] // 2 - self.scaled_resolution[0] // 2,
                              self.screen_res[1] // 2 - self.scaled_resolution[1] // 2]
        region = (self.screen_offset[0], self.screen_offset[1], self.scaled_resolution[0], self.scaled_resolution[1])
        img = pyautogui.screenshot('screencap.png', region=region)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        #self.img_visualization = img.copy()
        return img
    
    def pre_processing(self, img: cv2.Mat):
        """
        Pre-processes the image for object recognition by
        converting it to grayscale, blurring it, and then applying Canny edge detection.
        :param img: cv2.Mat
        :return: cv2.Mat
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_canny = cv2.Canny(img_blur, 50, 50)
        kernel = np.ones((5, 5))
        img_dial = cv2.dilate(img_canny, kernel, iterations=2)
        img_thres = cv2.erode(img_dial, kernel, iterations=1)
        return img_thres

    def object_recognition(self, img: cv2.Mat, verbose: bool):
        """
        Performs object recognition on the given pre-processed image
        :param img: cv2.Mat
        :return: Dictionary of arrays of detected objects
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width = img.shape[:2]
        mask_result = np.zeros((height, width), dtype=np.uint8)

        # BGR Blacklist
        for mask_setting in self.masks["BGR"]["blacklist"]:
            print(mask_setting)
            mask = cv2.inRange(img, np.array(mask_setting["lower"]), np.array(mask_setting["upper"]))
            mask_result = cv2.bitwise_or(mask_result, mask)
        mask_result = cv2.bitwise_not(mask_result)  # Invert mask to keep everything except the blacklist

        # BGR Whitelist
        for mask_setting in self.masks["BGR"]["whitelist"]:
            mask = cv2.inRange(img, np.array(mask_setting["lower"]), np.array(mask_setting["upper"]))
            mask_result = cv2.bitwise_or(mask_result, mask)

        masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_result)

        self.img_visualization = masked_img.copy()

        objects = self.parse_contours(self.pre_processing(masked_img), "Object", verbose)
        return objects

    def parse_contours(self, img: cv2.Mat, mask_setting: str, verbose: bool):
        """
        Extracts objects from the image using contour detection
        :param img: cv2.Mat
        :param mask_setting: string indicating which mask was used
        :param verbose: bool
        :return: array of dictionaries
        """
        objects = []
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:  # Iterate through each detected object
            area = cv2.contourArea(cnt)  # Get the area of the object
            if verbose:
                print(f"Detected contour with area {area}")
            if area > 800:  # Filter out small objects
                perimeter = cv2.arcLength(cnt, True)  # Calculate object's perimeter
                circularity = 4 * np.pi * (area / (perimeter ** 2))  # Calculate how circular the object is
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # Approximate the object's shape
                x, y, w, h = cv2.boundingRect(approx)  # Get the object's bounding box
                origin = (x + (w // 2) + self.screen_offset[0],
                          y + (h // 2) + self.screen_offset[1])  # Absolute screen pos of the center of the object

                object_type = mask_setting
                if circularity > 0.7:
                    object_type = "circle"

                # Draw rectangle and label over object for visualization
                cv2.rectangle(self.img_visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.img_visualization, object_type,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 127, 0), 2)

                objects.append({"obj": object_type, "pos": origin, "area": area})
        return objects

    def show_visual(self):
        """
        Shows a window with an image of what the AI sees for visualization
        :return:
        """
        if self.img_visualization is None:
            return
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", self.scaled_resolution[0], self.scaled_resolution[1])  # set initial size
        cv2.imshow("Processed Image", self.img_visualization)
        cv2.waitKey(1)
