import logging
import cv2

class Config:
    def __init__(self):
        self.camera_index = 0
        self.camera_api = cv2.CAP_DSHOW
        self.output_width = 1280
        self.output_height = 720
        self.max_fps = 30
        self.max_eye_distance = 30
        self.logging_level = logging.DEBUG
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
