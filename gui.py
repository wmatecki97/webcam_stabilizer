import customtkinter
import json
import logging
import cv2
import threading
import os
from camera_passthrough import run_camera

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Camera Config App")
        self.geometry("600x400")

        # Set the icon
        icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
        else:
            logging.warning(f"Icon file not found at: {icon_path}")


        self.config = self.load_config()
        self.camera_running = threading.Event()
        self.camera_thread = None

        # Camera Index
        self.camera_index_frame = customtkinter.CTkFrame(self)
        self.camera_index_frame.pack(padx=10, pady=10, fill="x")
        self.camera_index_label = customtkinter.CTkLabel(self.camera_index_frame, text="Camera Index:")
        self.camera_index_label.pack(side="left", padx=10)
        self.camera_index_entry = customtkinter.CTkEntry(self.camera_index_frame, width=50)
        self.camera_index_entry.pack(side="left", padx=10)
        self.camera_index_entry.insert(0, str(self.config.get("camera_index", 0)))

        # Output Width
        self.output_width_frame = customtkinter.CTkFrame(self)
        self.output_width_frame.pack(padx=10, pady=10, fill="x")
        self.output_width_label = customtkinter.CTkLabel(self.output_width_frame, text="Output Width:")
        self.output_width_label.pack(side="left", padx=10)
        self.output_width_entry = customtkinter.CTkEntry(self.output_width_frame, width=50)
        self.output_width_entry.pack(side="left", padx=10)
        self.output_width_entry.insert(0, str(self.config.get("output_width", 1280)))

        # Output Height
        self.output_height_frame = customtkinter.CTkFrame(self)
        self.output_height_frame.pack(padx=10, pady=10, fill="x")
        self.output_height_label = customtkinter.CTkLabel(self.output_height_frame, text="Output Height:")
        self.output_height_label.pack(side="left", padx=10)
        self.output_height_entry = customtkinter.CTkEntry(self.output_height_frame, width=50)
        self.output_height_entry.pack(side="left", padx=10)
        self.output_height_entry.insert(0, str(self.config.get("output_height", 720)))

        # Horizontal Alignment
        self.horizontal_frame = customtkinter.CTkFrame(self)
        self.horizontal_frame.pack(padx=10, pady=10, fill="x")
        self.horizontal_label = customtkinter.CTkLabel(self.horizontal_frame, text="Horizontal Alignment:")
        self.horizontal_label.pack(side="left", padx=2)
        self.horizontal_left_label = customtkinter.CTkLabel(self.horizontal_frame, text="Left")
        self.horizontal_left_label.pack(side="left", padx=2)
        self.horizontal_slider = customtkinter.CTkSlider(self.horizontal_frame, from_=0, to=100, width=200)
        self.horizontal_slider.pack(side="left", padx=2)
        self.horizontal_slider.set(self.config.get("horizontal", 50))
        self.horizontal_right_label = customtkinter.CTkLabel(self.horizontal_frame, text="Right")
        self.horizontal_right_label.pack(side="left", padx=2)

        # Vertical Alignment
        self.vertical_frame = customtkinter.CTkFrame(self)
        self.vertical_frame.pack(padx=10, pady=10, fill="x")
        self.vertical_label = customtkinter.CTkLabel(self.vertical_frame, text="Vertical Alignment:")
        self.vertical_label.pack(side="left", padx=2)
        self.vertical_slider = customtkinter.CTkSlider(self.vertical_frame, from_=0, to=100, width=200)
        self.vertical_slider.pack(side="left", padx=2)
        self.vertical_slider.set(self.config.get("vertical", 50))
        self.vertical_up_label = customtkinter.CTkLabel(self.vertical_frame, text="Up")
        self.vertical_up_label.pack(side="left", padx=5)
        self.vertical_down_label = customtkinter.CTkLabel(self.vertical_frame, text="Down")
        self.vertical_down_label.pack(side="left", padx=5)

        # Start Button
        self.start_button = customtkinter.CTkButton(self, text="Start Camera", command=self.start_camera)
        self.start_button.pack(padx=10, pady=20, fill="x")

        # Stop Button
        self.stop_button = customtkinter.CTkButton(self, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.pack(padx=10, pady=10, fill="x")

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
        if not self.camera_running.is_set():
            self.camera_running.set()
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            
            # Update config with current values from GUI
            self.config["camera_index"] = int(self.camera_index_entry.get())
            self.config["output_width"] = int(self.output_width_entry.get())
            self.config["output_height"] = int(self.output_height_entry.get())
            self.config["horizontal"] = int(self.horizontal_slider.get())
            self.config["vertical"] = int(self.vertical_slider.get())
            self.save_config()

            from config import Config
            config_obj = Config()
            config_obj.camera_index = self.config["camera_index"]
            config_obj.output_width = self.config["output_width"]
            config_obj.output_height = self.config["output_height"]
            config_obj.horizontal = self.config["horizontal"]
            config_obj.vertical = self.config["vertical"]

            self.camera_thread = threading.Thread(target=self.run_camera_thread, args=(config_obj,))
            self.camera_thread.start()

    def run_camera_thread(self, config):
        run_camera(config, self.camera_running)
        self.camera_running.clear()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def stop_camera(self):
        if self.camera_running.is_set():
            self.camera_running.clear()
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def on_closing(self):
        self.stop_camera()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join()
        self.destroy()

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")
    app = App()
    app.mainloop()
