import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import socket
import json
import time
import numpy as np

class PatternApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 1920
        self.canvas_height = 1200
        self.display_width = 960
        self.display_height = 600

        # Pattern state
        self.spot_radius = 30
        self.spot_center = [1148,916]# [self.canvas_width // 2, self.canvas_height // 2]
        self.grating_spacing = 6
        self.grating_size = 300
        self.grating_center = [1148,916]# [self.canvas_width // 2, self.canvas_height // 2]
        self.angle_deg = 34.5
        self.mode = "spot"
        self.keys_pressed = set()

        # Connect to display server
        self.server_ip = "192.168.1.102"
        self.server_port = 5000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.server_ip, self.server_port))
            print("Connected to display server.")
        except Exception as e:
            print("Failed to connect to server")

            print(e)
            
            self.sock = None

        # UI setup
        self.main_window = tk.Toplevel(self.root)
        self.main_window.title("Main - Control Window")
        self.canvas1 = tk.Canvas(self.main_window, width=self.display_width, height=self.display_height)
        self.canvas1.pack()
        self.main_window.protocol("WM_DELETE_WINDOW", self.quit_program)
        self.main_window.bind("<KeyPress>", self.key_down)
        self.main_window.bind("<KeyRelease>", self.key_up)
        self.main_window.focus_set()

        # self.mirror_window = tk.Toplevel(self.root)
        # self.mirror_window.title("Mirror - Display Window")
        # self.canvas2 = tk.Canvas(self.mirror_window, width=self.display_width, height=self.display_height)
        # self.canvas2.pack()
        # self.mirror_window.protocol("WM_DELETE_WINDOW", self.quit_program)

        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.tk_img = None

        self.img_on_canvas1 = self.canvas1.create_image(0, 0, anchor=tk.NW)
        # self.img_on_canvas2 = self.canvas2.create_image(0, 0, anchor=tk.NW)

        self.dragging = False
        self.canvas1.bind("<ButtonPress-1>", self.start_drag)
        self.canvas1.bind("<B1-Motion>", self.do_drag)
        self.canvas1.bind("<ButtonRelease-1>", self.stop_drag)
        self.canvas1.bind("<MouseWheel>", self.resize_pattern)
        self.canvas1.bind("<Button-4>", self.resize_pattern)
        self.canvas1.bind("<Button-5>", self.resize_pattern)

        self.update_image()

    def quit_program(self):
        self.root.quit()

    def send_update(self):
        if not self.sock:
            return
        try:
            if self.mode == "spot":
                data = {
                    "mask": "spot",
                    "center": self.spot_center,
                    "dimension": self.spot_radius,
                    "spacing":self.grating_spacing,
                    "angle":self.angle_deg,
                    "phase": 3.14                    
                }
            else:
                data = {
                    "mask": "grating",
                    "center": self.grating_center,
                    "dimension": self.grating_size,
                    "spacing":self.grating_spacing,
                    "angle":self.angle_deg,
                    "phase": 3.14
                }
            message = json.dumps(data).encode() + b'\n'
            time.sleep(0.02)
            self.sock.sendall(message)
        except Exception as e:
            print("Error sending data:", e)
# I found some gen AIs are stupid. For example, they cannot read a pdf file and give me the latex form directly. Instead, they just work well when I feed them page by page. How can I make a (python) code, which can read pdf file and sent them to a gen AI to generate latex?   
    def update_title(self):
        center = self.spot_center if self.mode == "spot" else self.grating_center
        self.main_window.title(f"Main - Control Window | Mode: {self.mode} | Center: ({center[0]}, {center[1]}) | Size: {self.spot_radius} | Spacing: {self.grating_spacing} | Angle: {self.angle_deg}")

    def update_image(self):
        self.image.paste("white", [0, 0, self.canvas_width, self.canvas_height])
        if self.mode == "spot":
            x, y = self.spot_center
            r = self.spot_radius
            self.draw.ellipse((x - r, y - r, x + r, y + r), fill="black", outline="white")
        else:
            cx, cy = self.grating_center
            half = self.grating_size // 2
            period = self.grating_spacing
            bar_width = int(period * 0.5)
            x0 = cx - half
            y0 = cy - half
            x1 = cx + half
            y1 = cy + half

            angle = getattr(self, "angle_deg", 0)  # rotation in degrees
            theta = np.deg2rad(angle)
            c, s = np.cos(theta), np.sin(theta)

            # loop over ROI and fill rotated grating
            for y in range(y0, y1):
                for x in range(x0, x1):
                    # project point onto the grating axis
                    proj = (x - cx) * c + (y - cy) * s
                    if (proj % period) < bar_width:
                        self.draw.point((x, y), fill="black")

        resized = self.image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas1.itemconfig(self.img_on_canvas1, image=self.tk_img)
        # self.canvas2.itemconfig(self.img_on_canvas2, image=self.tk_img)
        self.canvas1.image = self.tk_img
        # self.canvas2.image = self.tk_img
        self.update_title()

    def start_drag(self, event):
        self.dragging = True
        center = self.spot_center if self.mode == "spot" else self.grating_center
        self.drag_offset = (center[0] - event.x * 2, center[1] - event.y * 2)

    def do_drag(self, event):
        if not self.dragging:
            return
        if self.mode == "spot":
            self.spot_center = [event.x * 2 + self.drag_offset[0], event.y * 2 + self.drag_offset[1]]
        else:
            self.grating_center = [event.x * 2 + self.drag_offset[0], event.y * 2 + self.drag_offset[1]]
        self.update_image()
        self.send_update()

    def stop_drag(self, event):
        self.dragging = False

    def resize_pattern(self, event):
        delta = 1 if event.delta > 0 or event.num == 4 else -1
        if self.mode == "spot":
            self.spot_radius = max(5, self.spot_radius + delta * 2)
        else:
            self.grating_size = max(2, self.grating_size + delta)
        self.update_image()
        self.send_update()

    def key_down(self, event):
        self.keys_pressed.add(event.keysym)
        self.check_combination(event.keysym)

    def key_up(self, event):
        self.keys_pressed.discard(event.keysym)

    def check_combination(self, key):
        ctrl = "Control_L" in self.keys_pressed or "Control_R" in self.keys_pressed
        shift = "Shift_L" in self.keys_pressed or "Shift_R" in self.keys_pressed
        step = 1 if ctrl else 5

        if key == "Escape":
            self.mode = "spot"
            self.spot_radius = 0
            self.quit_program()
        elif key == "1":
            if self.mode != "spot":
                self.spot_center = self.grating_center[:]
            self.mode = "spot"
        elif key == "2":
            if self.mode != "grating":
                self.grating_center = self.spot_center[:]
            self.mode = "grating"
        elif key == "0":
            if self.mode != "spot":
                self.spot_center = self.spot_center[:]
            self.mode = "spot"
            self.spot_radius = 0 
        elif key == "Left":
            if self.mode == "spot":
                self.spot_center[0] -= step
            else:
                self.grating_center[0] -= step
        elif key == "Right":
            if self.mode == "spot":
                self.spot_center[0] += step
            else:
                self.grating_center[0] += step
        elif key == "Up":
            if self.mode == "spot":
                self.spot_center[1] -= step
            else:
                self.grating_center[1] -= step
        elif key == "Down":
            if self.mode == "spot":
                self.spot_center[1] += step
            else:
                self.grating_center[1] += step
        elif key in ["plus", "equal"]:
            if self.mode == "spot":
                self.spot_radius += 1 if ctrl else 5
            else:
                if shift:
                    self.grating_spacing += 1 
                else:
                    self.grating_size += 1 if ctrl else 5
        elif key in ["minus", "underscore"]:
            if self.mode == "spot":
                self.spot_radius = max(2, self.spot_radius - (1 if ctrl else 5))
            else:
                if shift:
                    self.grating_spacing = max(2, self.grating_spacing - 1)
                else:
                    self.grating_size = max(2, self.grating_size - (1 if ctrl else 5))
        elif key == "space":
            if shift:
                self.angle_deg -= 0.5
            else:
                self.angle_deg += 0.5

        self.update_image()
        self.send_update()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = PatternApp(root)
    root.mainloop()
