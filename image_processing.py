import cv2
import numpy as np
import json
import pytesseract
import geometry_utils
import time


class ImageProcessing:
    def __init__(self):
        self.masks = {}
        with open("masks.json", encoding='utf-8') as file:
            self.masks = json.load(file)

        self.cluster_settings = {}
        with open("cluster_settings.json", encoding='utf-8') as file:
            self.cluster_settings = json.load(file)

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

    def object_recognition(self, img: cv2.Mat, visualize: bool, verbose: bool):
        """
        Performs object recognition on the given pre-processed image
        :param img: cv2.Mat
        :param visualize: whether to visualize the image
        :param verbose: bool
        :return: List of detected GameObjects
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
                start_time = time.time()
                objects.extend(
                    self.parse_contours_to_objects(self.pre_processing(masked_imgs[mask_type]), label=mask_type,
                                                   verbose=verbose))
                if verbose:
                    print(f"Object recognition for {mask_type} took {time.time() - start_time:.2f} seconds")

                # Remove objects from general image that are in masked_imgs[mask_type] so we dont overwrite them
                mask_result = cv2.bitwise_and(mask_result, cv2.bitwise_not(result))

                if visualize:
                    # Visualize mask
                    cv2.namedWindow(f"{mask_type} mask", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(f"{mask_type} mask", 1200, 800)
                    cv2.imshow(f"{mask_type} mask", masked_imgs[mask_type])
        masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_result)
        start_time = time.time()
        objects.extend(
            self.parse_contours_to_objects(self.pre_processing(masked_img), label="unknown", verbose=verbose))
        if verbose:
            print(f"Object recognition for unknown took {time.time() - start_time:.2f} seconds")

        if visualize:
            # Visualize default object mask
            cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Masked Image", 1200, 800)
            cv2.imshow("Masked Image", masked_img)

            # Visualize objects
            for obj in objects:
                # Draw rectangle and label over object for visualization
                x1 = int(obj.bounding_box[0].x)
                y1 = int(obj.bounding_box[0].y)
                x2 = int(obj.bounding_box[1].x)
                y2 = int(obj.bounding_box[1].y)
                cv2.rectangle(self.img_visualization,
                              pt1=(x1, y1),
                              pt2=(x2, y2),
                              color=(0, 255, 0), thickness=2)
                cv2.putText(self.img_visualization, text=f"{obj.label} n={obj.count}",
                            org=(x1, y1 - 10),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                            color=(0, 255, 0), thickness=2)
                cv2.putText(self.img_visualization,
                            text=f"A={obj.area:.2f} P={obj.perimeter:.2f} C={obj.circularity:.2f}",
                            org=(x1, y2 + 15),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                            color=(0, 255, 0), thickness=2)
            self.show_visual(True)

        if verbose:
            print(f"Identified {len(objects)} objects")

        return objects

    @staticmethod
    def cluster_or_add(obj: geometry_utils.GameObject, objects: list[geometry_utils.GameObject], label: str, max_count: int, cluster_by: str, cluster_distance: float):
        """
        Cluster 'obj' with any nearby cluster of the same type or add it to 'objects' if no clusters found
        :param obj: object to cluster or add to objects
        :param objects: all possible GOs to cluster with
        :param label: type to cluster with
        :param max_count: maximum number of objects in a cluster
        :param cluster_by: cluster by distance between "center" or "edge"
        :param cluster_distance: maximum distance to cluster by
        :return:
        """
        if max_count > 1:
            # Try merging this object with any nearby cluster
            for other_obj in objects:
                if other_obj.count >= max_count:
                    continue
                if other_obj.label != label:
                    continue

                # Cluster by center by default
                dx = obj.pos.x - other_obj.pos.x
                dy = obj.pos.y - other_obj.pos.y
                if cluster_by == "edge":
                    # Cluster by distance between closest edges of bounding boxes
                    dx, dy = obj.linear_bounds_distance(other_obj.bounding_box)

                if dx * dx + dy * dy < cluster_distance * cluster_distance:
                    # Update position as weighted average by area
                    # (x, y) = (A1x1 + A2x2) / (A1+A2), (A1y1 + A2y2) / (A1+A2)
                    total_area = other_obj.area + obj.area
                    other_obj.pos = geometry_utils.Vector(
                        int((other_obj.area * other_obj.pos.x + obj.area * obj.pos.x) / total_area),
                        int((other_obj.area * other_obj.pos.y + obj.area * obj.pos.y) / total_area))

                    other_obj.area = total_area
                    other_obj.perimeter += obj.perimeter
                    other_obj.count += 1
                    other_obj.circularity = (other_obj.circularity + obj.circularity) / 2
                    other_obj.extend_bounds(obj.bounding_box)
                    break
                # No need to append this obj to objects list since we merged it with other_obj
            else:
                # No clusters found so add to objects list
                objects.append(obj)
        else:
            # Don't waste time making clusters of 1 object
            objects.append(obj)

    def parse_contours_to_objects(self, img: cv2.Mat, label: str, verbose: bool):
        """
        Extracts objects from the image using contour detection
        :param img: cv2.Mat
        :param label: label to assign to objects and key to find cluster settings with
        :param verbose: bool
        :return: List of GameObjects
        """
        # Cache settings outside the loop to avoid repeated dict lookups
        settings = self.cluster_settings.get(label, {})
        identify_by = settings.get("identify_by")
        cluster_by = settings.get("cluster_by", "center")
        max_count = settings.get("max_count", 1)
        cluster_distance = settings.get("cluster_distance", 0)
        variants = settings.get("variants", None)

        img_center_x = img.shape[1] / 2
        img_center_y = img.shape[0] / 2

        objects = []
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)  # Get the area of the contour
            if area < 100:  # Filter out small noise
                continue

            perimeter = cv2.arcLength(cnt, True)  # Calculate object's perimeter
            if perimeter == 0:
                continue  # Avoid divide-by-zero

            circularity = 4 * np.pi * (area / (perimeter ** 2))  # Calculate how circular the object is
            x, y, w, h = cv2.boundingRect(cnt)  # Get the object's bounding box
            half_w = w * 0.5
            half_h = h * 0.5
            origin = geometry_utils.Vector(x + half_w - img_center_x,
                                           y + half_h - img_center_y)  # Position of the center of the object relative to the image center

            if verbose:
                print(f"Detected contour at {origin}: A={area}, P={perimeter}, C={circularity}")

            if variants is not None:
                if identify_by == "perimeter":
                    # Check variants for matching perimeter, this is used later to cluster objects of the same type
                    for key, value in variants.items():
                        if value["min_perimeter"] < perimeter < value["max_perimeter"]:
                            label = key
                            cluster_by = value.get("cluster_by", cluster_by)
                            max_count = value.get("max_count", max_count)
                            cluster_distance = value.get("cluster_distance", cluster_distance)
                            break

            obj = geometry_utils.GameObject(label=label,
                                            pos=origin,
                                            perimeter=perimeter,
                                            area=area,
                                            circularity=circularity,
                                            count=1,
                                            bounding_box=(geometry_utils.Vector(x, y), geometry_utils.Vector(x + w, y + h)))

            self.cluster_or_add(obj, objects, label, max_count, cluster_by, cluster_distance)
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
        cv2.resizeWindow("Processed Image", 1200, 800)
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
