import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

class PatternApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 960
        self.canvas_height = 600

        # Spot and grating parameters
        self.spot_radius = 30
        self.spot_center = [self.canvas_width // 2, self.canvas_height // 2]

        self.grating_spacing = 10  # period
        self.grating_size = 50
        self.grating_center = [self.canvas_width // 2, self.canvas_height // 2]

        self.mode = "spot"  # 'spot' or 'grating'

        self.keys_pressed = set()

        # Main window
        self.main_window = tk.Toplevel(self.root)
        self.canvas1 = tk.Canvas(self.main_window, width=self.canvas_width, height=self.canvas_height)
        self.canvas1.pack()
        self.main_window.protocol("WM_DELETE_WINDOW", self.quit_program)
        self.main_window.bind("<KeyPress>", self.key_down)
        self.main_window.bind("<KeyRelease>", self.key_up)
        self.main_window.focus_set()

        # Mirror window
        self.mirror_window = tk.Toplevel(self.root)
        self.canvas2 = tk.Canvas(self.mirror_window, width=self.canvas_width, height=self.canvas_height)
        self.canvas2.pack()
        self.mirror_window.protocol("WM_DELETE_WINDOW", self.quit_program)
        
        self.mirror_window.title(f"Mirror - Display Window")
        # Image buffer
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.tk_img = ImageTk.PhotoImage(self.image)

        # Image display
        self.img_on_canvas1 = self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.img_on_canvas2 = self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # Mouse interactions
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

    def update_title(self):
        if self.mode == "spot":
            x, y = self.spot_center
        else:
            x, y = self.grating_center
        self.main_window.title(f"Main - Control Window | Mode: {self.mode} | Center: ({x}, {y})")

    def update_image(self):
        self.image.paste("white", [0, 0, self.canvas_width, self.canvas_height])
        if self.mode == "spot":
            x, y = self.spot_center
            r = self.spot_radius
            self.draw.ellipse((x - r, y - r, x + r, y + r), fill="blue", outline="black")
        else:
            cx, cy = self.grating_center
            half = self.grating_size // 2
            period = self.grating_spacing
            bar_width = int(period * 0.5)
            x0 = cx - half
            y0 = cy - half
            x1 = cx + half
            y1 = cy + half
            for x in range(x0, x1 + 1, period):
                self.draw.rectangle([x, y0, x + bar_width - 1, y1], fill="black")

        self.tk_img = ImageTk.PhotoImage(self.image)
        self.canvas1.itemconfig(self.img_on_canvas1, image=self.tk_img)
        self.canvas2.itemconfig(self.img_on_canvas2, image=self.tk_img)
        self.canvas1.image = self.tk_img
        self.canvas2.image = self.tk_img
        self.update_title()

    def start_drag(self, event):
        self.dragging = True
        center = self.spot_center if self.mode == "spot" else self.grating_center
        self.drag_offset = (center[0] - event.x, center[1] - event.y)

    def do_drag(self, event):
        if not self.dragging:
            return
        if self.mode == "spot":
            self.spot_center = [event.x + self.drag_offset[0], event.y + self.drag_offset[1]]
        else:
            self.grating_center = [event.x + self.drag_offset[0], event.y + self.drag_offset[1]]
        self.update_image()

    def stop_drag(self, event):
        self.dragging = False

    def resize_pattern(self, event):
        delta = 1 if event.delta > 0 or event.num == 4 else -1
        if self.mode == "spot":
            self.spot_radius = max(5, self.spot_radius + delta * 2)
        else:
            self.grating_spacing = max(2, self.grating_spacing + delta)
        self.update_image()

    def key_down(self, event):
        self.keys_pressed.add(event.keysym)
        self.check_combination(event.keysym)

    def key_up(self, event):
        self.keys_pressed.discard(event.keysym)

    def check_combination(self, key):
        ctrl = "Control_L" in self.keys_pressed or "Control_R" in self.keys_pressed
        step = 1 if ctrl else 5

        if key == "Escape":
            self.quit_program()
        elif key == "1":
            if self.mode != "spot":
                self.spot_center = self.grating_center[:]
            self.mode = "spot"
        elif key == "2":
            if self.mode != "grating":
                self.grating_center = self.spot_center[:]
            self.mode = "grating"
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
                self.grating_spacing += 1
        elif key == "minus":
            if self.mode == "spot":
                self.spot_radius = max(5, self.spot_radius - (1 if ctrl else 5))
            else:
                self.grating_spacing = max(2, self.grating_spacing - 1)

        self.update_image()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = PatternApp(root)
    root.mainloop()