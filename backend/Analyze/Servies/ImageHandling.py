import cv2 as cv
import numpy as np
from PIL import Image, ImageOps

from Common.ConstantMeta import ConstantMeta


class ImageHandlingConstants(metaclass=ConstantMeta):
    MAX_SIZE = 1600
    TARGET_MARGIN = 0.15
    BLUR_KERNEL_SIZE = (5, 5)
    BLUR_SIGMA = 0
    EDGE_THRESHOLD_LOW = 100
    EDGE_THRESHOLD_HIGH = 200

    MIN_CONTOUR_POINTS = 20
    ELLIPSE_REFIT_ITERATIONS = 3
    ELLIPSE_INLIER_TOLERANCE = 0.12
    MIN_AXIS_LENGTH = 20
    MAX_AXIS_RATIO = 3
    MIN_AXES_PRODUCT_RATIO = 0.01
    MAX_AXIS_IMAGE_RATIO = 1.5

    DISTANCE_MASK_SIZE = 3
    SECTOR_COUNT = 12
    SAMPLES_PER_SECTOR = 10
    ELLIPSE_SAMPLE_COUNT = SECTOR_COUNT * SAMPLES_PER_SECTOR
    MIN_EDGE_DISTANCE = 2
    EDGE_DISTANCE_AXIS_RATIO = 0.005
    MIN_SECTOR_SUPPORT = 0.5
    MIN_SUPPORTED_SECTORS = 8

    INNER_RING_RATIO_MIN = 0.2
    INNER_RING_RATIO_MAX = 0.85
    MAX_RING_RATIO_DIFFERENCE = 0.15
    MAX_CENTER_GAP_RATIO = 0.12
    NEAR_CIRCULAR_AXIS_RATIO = 1.15
    MAX_RING_ANGLE_GAP = 15


class ImageHandling:
    def __init__(self):
        self.__image = None

    def preprocess(self, image_path: str):
        if not self.__load_image(image_path):
            return None
        self.__correct_orientation()
        self.__remove_noise()
        self.__detect_edges()
        candidates = self.__find_ellipse_candidates()
        self.__crop_target(candidates)
        self.__resize_image()
        return self.__image
    
    # Image Checking and Loading
    def __load_image(self, image_path: str) -> bool:
        try:
            with Image.open(image_path) as image:
                self.__image = image.copy()
            return True
        except (OSError, ValueError, cv.error):
            return False
        
    # Image Preprocessing Methods
    def __correct_orientation(self):        
        image = ImageOps.exif_transpose(self.__image).convert("RGB")
        self.__image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Image Blur and Noise Removal
    def __remove_noise(self):
        blur = cv.GaussianBlur(
            self.__image, 
            ImageHandlingConstants.BLUR_KERNEL_SIZE, 
            ImageHandlingConstants.BLUR_SIGMA
        )
        self.__image = blur

    # Edge Detection
    def __detect_edges(self):
        self.__image = cv.Canny(
            self.__image,
            ImageHandlingConstants.EDGE_THRESHOLD_LOW, 
            ImageHandlingConstants.EDGE_THRESHOLD_HIGH
        )

    # Ellipse Detection
    def __find_ellipse_candidates(self):
        contours, _ = cv.findContours(
            self.__image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            if len(contour) < ImageHandlingConstants.MIN_CONTOUR_POINTS:
                continue
            points = contour.reshape(-1, 2).astype(np.float32)
            for _ in range(ImageHandlingConstants.ELLIPSE_REFIT_ITERATIONS):
                ellipse = cv.fitEllipse(points.reshape(-1, 1, 2))
                center, axes, angle = ellipse
                if min(axes) <= 0:
                    break
                radians = np.deg2rad(angle)
                rotation = np.array([[np.cos(radians), -np.sin(radians)],
                                     [np.sin(radians), np.cos(radians)]])
                local = (points - center) @ rotation
                radius = np.linalg.norm(local / (np.array(axes) / 2), axis=1)
                inliers = np.abs(radius - 1) < ImageHandlingConstants.ELLIPSE_INLIER_TOLERANCE
                if inliers.all() or np.count_nonzero(inliers) < ImageHandlingConstants.MIN_CONTOUR_POINTS:
                    break
                points = points[inliers]
            candidates.append(cv.fitEllipse(points.reshape(-1, 1, 2)))

        return candidates

    # Target Cropping
    def __crop_target(self, candidates):
        target = self.__select_target(candidates)
        if target is None:
            return

        center, axes, angle = target

        factor = 1 + ImageHandlingConstants.TARGET_MARGIN
        box = cv.boxPoints((
            center,
            (axes[0] * factor, axes[1] * factor),
            angle,
        ))

        height, width = self.__image.shape[:2]
        x1, y1 = np.clip(np.floor(box.min(axis=0)), [0, 0], [width, height]).astype(int)
        x2, y2 = np.clip(np.ceil(box.max(axis=0)), [0, 0], [width, height]).astype(int)

        if x2 > x1 and y2 > y1:
            self.__image = self.__image[y1:y2, x1:x2].copy()

    # Target Selection
    def __select_target(self, candidates):
        height, width = self.__image.shape[:2]
        distance = cv.distanceTransform(255 - self.__image, cv.DIST_L2, ImageHandlingConstants.DISTANCE_MASK_SIZE)
        valid = []
        for ellipse in candidates:
            (cx, cy), axes, angle = ellipse
            if not np.isfinite([cx, cy, *axes, angle]).all():
                continue
            if min(axes) < ImageHandlingConstants.MIN_AXIS_LENGTH or max(axes) / min(axes) > ImageHandlingConstants.MAX_AXIS_RATIO:
                continue
            if not (0 <= cx < width and 0 <= cy < height):
                continue
            if np.prod(axes) < width * height * ImageHandlingConstants.MIN_AXES_PRODUCT_RATIO or max(axes) > max(width, height) * ImageHandlingConstants.MAX_AXIS_IMAGE_RATIO:
                continue
            t = np.linspace(0, 2 * np.pi, ImageHandlingConstants.ELLIPSE_SAMPLE_COUNT, endpoint=False)
            local = np.column_stack((np.cos(t) * axes[0] / 2, np.sin(t) * axes[1] / 2))
            a = np.deg2rad(angle)
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            samples = np.rint(local @ rotation.T + (cx, cy)).astype(int)
            x, y = samples.T
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            supported = np.zeros(ImageHandlingConstants.ELLIPSE_SAMPLE_COUNT, dtype=bool)
            supported[inside] = distance[y[inside], x[inside]] <= max(ImageHandlingConstants.MIN_EDGE_DISTANCE, min(axes) * ImageHandlingConstants.EDGE_DISTANCE_AXIS_RATIO)
            
            if np.count_nonzero(supported.reshape(ImageHandlingConstants.SECTOR_COUNT, ImageHandlingConstants.SAMPLES_PER_SECTOR).mean(axis=1) >= ImageHandlingConstants.MIN_SECTOR_SUPPORT) >= ImageHandlingConstants.MIN_SUPPORTED_SECTORS:
                valid.append(ellipse)

        for outer in sorted(valid, key=lambda e: np.prod(e[1]), reverse=True):
            center, axes, angle = outer
            for inner_center, inner_axes, inner_angle in valid:
                ratios = np.sort(inner_axes) / np.sort(axes)
                center_gap = np.linalg.norm(np.array(center) - inner_center)
                angle_gap = abs(angle - inner_angle) % 180
                if (np.all((ratios > ImageHandlingConstants.INNER_RING_RATIO_MIN) & (ratios < ImageHandlingConstants.INNER_RING_RATIO_MAX))
                        and abs(ratios[0] - ratios[1]) < ImageHandlingConstants.MAX_RING_RATIO_DIFFERENCE
                        and center_gap < min(axes) * ImageHandlingConstants.MAX_CENTER_GAP_RATIO
                        and (max(axes) / min(axes) < ImageHandlingConstants.NEAR_CIRCULAR_AXIS_RATIO or min(angle_gap, 180 - angle_gap) < ImageHandlingConstants.MAX_RING_ANGLE_GAP)):
                    return outer
        return None
    
    # Image Resizing 
    def __resize_image(self):
        height, width = self.__image.shape[:2]
        scale = min(1.0, ImageHandlingConstants.MAX_SIZE / max(width, height))
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        self.__image = cv.resize(self.__image, (new_width, new_height))
