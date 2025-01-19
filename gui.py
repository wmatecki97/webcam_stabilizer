import customtkinter
import json
import logging
import cv2
import threading
from camera_passthrough import run_camera

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Camera Config App")
        self.geometry("600x400")

        self.config = self.load_config()
        self.camera_running = False
        self.camera_thread = None

        # Camera Index
        self.camera_index_label = customtkinter.CTkLabel(self, text="Camera Index:")
        self.camera_index_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.camera_index_entry = customtkinter.CTkEntry(self, width=50)
        self.camera_index_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.camera_index_entry.insert(0, str(self.config.get("camera_index", 0)))

        # Output Width
        self.output_width_label = customtkinter.CTkLabel(self, text="Output Width:")
        self.output_width_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.output_width_entry = customtkinter.CTkEntry(self, width=50)
        self.output_width_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.output_width_entry.insert(0, str(self.config.get("output_width", 1280)))

        # Output Height
        self.output_height_label = customtkinter.CTkLabel(self, text="Output Height:")
        self.output_height_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.output_height_entry = customtkinter.CTkEntry(self, width=50)
        self.output_height_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        self.output_height_entry.insert(0, str(self.config.get("output_height", 720)))

        # Horizontal Alignment
        self.horizontal_label = customtkinter.CTkLabel(self, text="Horizontal Alignment:")
        self.horizontal_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.horizontal_slider = customtkinter.CTkSlider(self, from_=0, to=100, width=200)
        self.horizontal_slider.grid(row=3, column=1, padx=10, pady=10, sticky="w")
        self.horizontal_slider.set(self.config.get("horizontal", 50))

        # Vertical Alignment
        self.vertical_label = customtkinter.CTkLabel(self, text="Vertical Alignment:")
        self.vertical_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.vertical_slider = customtkinter.CTkSlider(self, from_=0, to=100, width=200)
        self.vertical_slider.grid(row=4, column=1, padx=10, pady=10, sticky="w")
        self.vertical_slider.set(self.config.get("vertical", 50))

        # Start Button
        self.start_button = customtkinter.CTkButton(self, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

        # Stop Button
        self.stop_button = customtkinter.CTkButton(self, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_config(self):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_config(self):
        config = {
            "camera_index": int(self.camera_index_entry.get()),
            "output_width": int(self.output_width_entry.get()),
            "output_height": int(self.output_height_entry.get()),
            "horizontal": int(self.horizontal_slider.get()),
            "vertical": int(self.vertical_slider.get()),
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        return config

    def start_camera(self):
        if not self.camera_running:
            self.camera_running = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            config = self.save_config()
            self.camera_thread = threading.Thread(target=self.run_camera_thread, args=(config,))
            self.camera_thread.start()

    def run_camera_thread(self, config):
        class Config:
            def __init__(self, config_dict):
                self.camera_index = config_dict.get("camera_index", 0)
                self.camera_api = cv2.CAP_DSHOW
                self.output_width = config_dict.get("output_width", 1280)
                self.output_height = config_dict.get("output_height", 720)
                self.max_fps = 30
                self.logging_level = logging.ERROR
                self.log_format = '%(asctime)s - %(levelname)s - %(message)s'
                self.horizontal = config_dict.get("horizontal", 50)
                self.vertical = config_dict.get("vertical", 50)
        run_camera(Config(config))
        self.camera_running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def stop_camera(self):
        if self.camera_running:
            self.camera_running = False
            # No need to explicitly stop the thread, it will exit when camera_running is False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")
    app = App()
    app.mainloop()
