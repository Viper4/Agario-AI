import cv2
import numpy as np
import json
import pytesseract
from PIL import Image


class ImageProcessing:
    def __init__(self):
        self.masks = {}
        with open("masks.json", encoding='utf-8') as file:
            self.masks = json.load(file)

        self.img_visualization = None  # Image for visualizing processing results

    def convert_to_mat(self, png_bytes: bytes):
        """
        Converts bytes of a PNG image into a cv2.Mat.
        :param png_bytes: bytes of a PNG image
        :return: cv2.Mat
        """
        img_array = np.frombuffer(png_bytes, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

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
        :param verbose: bool
        :return: List of detected objects
        """
        self.img_visualization = img.copy()

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width = img.shape[:2]
        mask_result = np.zeros((height, width), dtype=np.uint8)

        masked_imgs = {}
        objects = []

        for mask_type in self.masks["BGR"]:
            if mask_type == "blacklist":  # Apply blacklist to entire image
                for mask_setting in self.masks["BGR"]["blacklist"]:
                    mask = cv2.inRange(img, np.array(mask_setting["lower"]), np.array(mask_setting["upper"]))
                    mask_result = cv2.bitwise_or(mask_result, mask)
                mask_result = cv2.bitwise_not(mask_result)  # Invert mask to keep everything except the blacklist
            elif mask_type == "whitelist":  # Apply whitelist to entire image
                for mask_setting in self.masks["BGR"]["whitelist"]:
                    mask = cv2.inRange(img, np.array(mask_setting["lower"]), np.array(mask_setting["upper"]))
                    mask_result = cv2.bitwise_or(mask_result, mask)
            else:  # Apply mask to identify specific things
                result = np.zeros((height, width), dtype=np.uint8)
                for mask_setting in self.masks["BGR"][mask_type]:
                    mask = cv2.inRange(img, np.array(mask_setting["lower"]), np.array(mask_setting["upper"]))
                    kernel = np.ones((5, 5), np.uint8)
                    dilated_mask = cv2.dilate(mask, kernel, iterations=4)  # Dilate mask to remove noise and edges
                    result = np.bitwise_or(result, dilated_mask)

                masked_imgs[mask_type] = cv2.bitwise_and(img_hsv, img_hsv, mask=result)
                objects.extend(self.parse_contours(self.pre_processing(masked_imgs[mask_type]), mask_type, verbose))
                # Remove objects from general image that are in masked_imgs[mask_type] so we dont overwrite them
                mask_result = cv2.bitwise_and(mask_result, cv2.bitwise_not(result))

                # Visualize mask
                cv2.namedWindow(f"{mask_type} mask", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"{mask_type} mask", 800, 400)
                cv2.imshow(f"{mask_type} mask", masked_imgs[mask_type])

        # Visualize default object mask
        masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_result)
        cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Masked Image", 800, 400)
        cv2.imshow("Masked Image", masked_img)

        objects.extend(self.parse_contours(self.pre_processing(masked_img), "object", verbose))
        return objects

    def parse_contours(self, img: cv2.Mat, object_label: str, verbose: bool):
        """
        Extracts objects from the image using contour detection
        :param img: cv2.Mat
        :param object_label: str
        :param verbose: bool
        :return: List of dictionaries
        """
        objects = []
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:  # Iterate through each detected object
            area = cv2.contourArea(cnt)  # Get the area of the object
            if verbose:
                print(f"Detected contour with area {area}")
            if area > 50:  # Filter out small objects
                perimeter = cv2.arcLength(cnt, True)  # Calculate object's perimeter
                circularity = 4 * np.pi * (area / (perimeter ** 2))  # Calculate how circular the object is
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # Approximate the object's shape
                x, y, w, h = cv2.boundingRect(approx)  # Get the object's bounding box
                origin = (x + (w // 2), y + (h // 2))  # Position of the center of the object relative to the image

                # Draw rectangle and label over object for visualization
                cv2.rectangle(self.img_visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.img_visualization, text=object_label + " " + str(len(contours)),
                            org=(x, y - 10),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8,
                            color=(0, 255, 0), thickness=2)

                objects.append({"obj": object_label, "pos": origin, "perimeter": perimeter,
                                "area": area, "circularity": circularity})
        return objects

    def show_visual(self, save_to_file: bool):
        """
        Shows a window with an image of what the AI sees for visualization
        :param save_to_file: If the visual image should be saved to a PNG file
        :return:
        """
        if self.img_visualization is None:
            return
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", 800, 400)
        cv2.imshow("Processed Image", self.img_visualization)
        if save_to_file:
            cv2.imwrite("processed_image.png", self.img_visualization)
        cv2.waitKey(1)
        
    def extract_text(self, png_bytes: bytes):
        """
        Run OCR on the image to extract text
        :param png_bytes: bytes
        :return: string
        """
        # Convert bytes to cv2.Mat
        img = self.convert_to_mat(png_bytes)

        # Convert to grayscale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Otsu thresholding to make OCR more accurate
        _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR
        text = pytesseract.image_to_string(binary_image, lang="eng")
        return text
