import logging
import cv2
from config import Config
from camera_passthrough import run_camera

def main():
    config = Config()
    logging.basicConfig(level=config.logging_level, format=config.log_format)
    run_camera(config)

if __name__ == "__main__":
    main()
