import logging
import cv2

class Config:
    def __init__(self):
        self.camera_index = 0
        self.camera_api = cv2.CAP_DSHOW
        self.output_width = 640
        self.output_height = 480
        self.max_fps = 60
        self.logging_level = logging.ERROR
        self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
        self.horizontal = 50
        self.vertical = 50
