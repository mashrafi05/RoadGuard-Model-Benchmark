"""
bbox_editor.py
==============
Bounding box editor for YOLO datasets.

Features
--------
  - Browse train / val / test splits
  - View all bounding boxes overlaid on image
  - Click a box to select it (highlighted in red)
  - Drag selected box to move it
  - Drag corners/edges to resize it
  - Press Delete to remove selected box
  - Draw new box by dragging on empty canvas
  - Choose class for new box from dropdown
  - All changes saved back to the .txt label file instantly
  - Keyboard navigation: Left/Right arrow to go prev/next image
  - Zoom: mouse wheel, or +/- keys

Requirements
------------
    pip install pillow

Run from I:/Project_Conference/
    python bbox_editor.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from PIL import Image, ImageTk
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT = Path("I:/Project_Conference")

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Speed Breaker",
}

# One distinct colour per class
CLASS_COLORS = {
    0: "#00C8FF",   # cyan
    1: "#FFD700",   # gold
    2: "#FF6B35",   # orange
    3: "#A855F7",   # purple
    4: "#22C55E",   # green
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HANDLE_SIZE = 8     # px — size of resize handles
MIN_BOX_PX  = 10    # minimum box side in pixels
# ─────────────────────────────────────────────────────────────────────────────


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """YOLO normalised → pixel coords (x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Pixel coords → YOLO normalised (cx,cy,w,h)."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def read_labels(lbl_path: Path):
    """Return list of [class_id, cx, cy, w, h] floats."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
            except ValueError:
                pass
    return boxes


def write_labels(lbl_path: Path, boxes: list):
    """Write boxes back to .txt file."""
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        cls_id, cx, cy, w, h = box
        # clamp to [0,1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    lbl_path.write_text("\n".join(lines), encoding="utf-8")


class BBoxEditor:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("YOLO Bounding Box Editor")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1280x800")

        # State
        self.images: list[Path] = []
        self.img_idx   = 0
        self.boxes     = []        # list of [cls_id, cx, cy, w, h]
        self.pil_image = None      # original PIL image
        self.tk_image  = None      # displayed ImageTk
        self.zoom      = 1.0
        self.pan_x     = 0
        self.pan_y     = 0
        self.canvas_w  = 1
        self.canvas_h  = 1
        self.img_w     = 1
        self.img_h     = 1

        # Interaction state
        self.selected_box  = None   # index into self.boxes
        self.drag_mode     = None   # "move" | "resize-NW" | "resize-NE" |
                                    # "resize-SE" | "resize-SW" | "draw"
        self.drag_start_x  = 0
        self.drag_start_y  = 0
        self.drag_orig_box = None   # snapshot of box before drag
        self.draw_rect_id  = None   # canvas item id while drawing

        self._build_ui()
        self._load_split("train")

    # ── UI CONSTRUCTION ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#16213e", pady=6)
        top.pack(fill=tk.X, side=tk.TOP)

        tk.Label(top, text="Split:", bg="#16213e", fg="#a0aec0",
                 font=("Consolas", 11)).pack(side=tk.LEFT, padx=(12, 4))
        self.split_var = tk.StringVar(value="train")
        split_cb = ttk.Combobox(top, textvariable=self.split_var,
                                values=["train", "val", "test"],
                                state="readonly", width=8)
        split_cb.pack(side=tk.LEFT, padx=4)
        split_cb.bind("<<ComboboxSelected>>",
                      lambda e: self._load_split(self.split_var.get()))

        tk.Button(top, text="◀  Prev", command=self.prev_image,
                  bg="#2d3561", fg="white", relief=tk.FLAT,
                  padx=10, font=("Consolas", 11)).pack(side=tk.LEFT, padx=8)

        self.img_counter = tk.Label(top, text="0 / 0", bg="#16213e",
                                    fg="#e2e8f0", font=("Consolas", 11),
                                    width=12)
        self.img_counter.pack(side=tk.LEFT)

        tk.Button(top, text="Next  ▶", command=self.next_image,
                  bg="#2d3561", fg="white", relief=tk.FLAT,
                  padx=10, font=("Consolas", 11)).pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="Open folder…", command=self._open_folder,
                  bg="#2d3561", fg="white", relief=tk.FLAT,
                  padx=10, font=("Consolas", 11)).pack(side=tk.LEFT, padx=8)

        self.filename_label = tk.Label(top, text="", bg="#16213e",
                                       fg="#68d391", font=("Consolas", 11))
        self.filename_label.pack(side=tk.LEFT, padx=12)

        # Zoom controls
        tk.Label(top, text="Zoom:", bg="#16213e", fg="#a0aec0",
                 font=("Consolas", 11)).pack(side=tk.RIGHT, padx=(0, 4))
        tk.Button(top, text="−", command=lambda: self._zoom(-0.2),
                  bg="#2d3561", fg="white", relief=tk.FLAT,
                  padx=8, font=("Consolas", 13)).pack(side=tk.RIGHT, padx=2)
        self.zoom_label = tk.Label(top, text="100%", bg="#16213e",
                                   fg="#e2e8f0", font=("Consolas", 11),
                                   width=5)
        self.zoom_label.pack(side=tk.RIGHT)
        tk.Button(top, text="+", command=lambda: self._zoom(0.2),
                  bg="#2d3561", fg="white", relief=tk.FLAT,
                  padx=8, font=("Consolas", 13)).pack(side=tk.RIGHT, padx=2)

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = tk.Canvas(main, bg="#0f0f1a", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>",   self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self.canvas.bind("<Button-4>",        self._on_mousewheel)
        self.canvas.bind("<Button-5>",        self._on_mousewheel)

        # Keyboard shortcuts
        self.root.bind("<Left>",   lambda e: self.prev_image())
        self.root.bind("<Right>",  lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected())
        self.root.bind("<plus>",   lambda e: self._zoom(0.2))
        self.root.bind("<minus>",  lambda e: self._zoom(-0.2))
        self.root.bind("<Escape>", lambda e: self._deselect())
        self.root.bind("s",        lambda e: self._save_current())

        # ── Right panel ───────────────────────────────────────────────────────
        right = tk.Frame(main, bg="#16213e", width=240)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # Class selector for new boxes
        tk.Label(right, text="NEW BOX CLASS", bg="#16213e", fg="#718096",
                 font=("Consolas", 9, "bold")).pack(pady=(14, 4), padx=12,
                                                      anchor="w")
        self.new_class_var = tk.IntVar(value=0)
        for cls_id, name in CLASS_NAMES.items():
            color = CLASS_COLORS.get(cls_id, "white")
            rb = tk.Radiobutton(right, text=f"{cls_id}: {name}",
                                variable=self.new_class_var, value=cls_id,
                                bg="#16213e", fg=color, selectcolor="#0f0f1a",
                                activebackground="#16213e",
                                font=("Consolas", 10))
            rb.pack(anchor="w", padx=14)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=12, pady=12)

        # Selected box info
        tk.Label(right, text="SELECTED BOX", bg="#16213e", fg="#718096",
                 font=("Consolas", 9, "bold")).pack(pady=(0, 4), padx=12,
                                                    anchor="w")
        self.sel_info = tk.Label(right, text="None", bg="#16213e",
                                 fg="#e2e8f0", font=("Consolas", 10),
                                 justify=tk.LEFT, wraplength=200)
        self.sel_info.pack(padx=14, anchor="w")

        # Class changer for selected box
        tk.Label(right, text="Change class:", bg="#16213e", fg="#718096",
                 font=("Consolas", 9)).pack(pady=(10, 2), padx=12, anchor="w")
        self.change_class_var = tk.IntVar(value=0)
        change_cb = ttk.Combobox(
            right,
            textvariable=self.change_class_var,
            values=[f"{k}: {v}" for k, v in CLASS_NAMES.items()],
            state="readonly", width=22)
        change_cb.pack(padx=12, anchor="w")
        change_cb.bind("<<ComboboxSelected>>", self._change_selected_class)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=12, pady=12)

        tk.Button(right, text="🗑  Delete selected  [Del]",
                  command=self.delete_selected,
                  bg="#c53030", fg="white", relief=tk.FLAT,
                  padx=8, pady=6, font=("Consolas", 10),
                  width=22).pack(padx=12)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=12, pady=12)

        # Box list
        tk.Label(right, text="ALL BOXES", bg="#16213e", fg="#718096",
                 font=("Consolas", 9, "bold")).pack(pady=(0, 4), padx=12,
                                                    anchor="w")
        self.box_listbox = tk.Listbox(right, bg="#0f0f1a", fg="#e2e8f0",
                                      selectbackground="#2d3561",
                                      font=("Consolas", 10), height=12,
                                      relief=tk.FLAT, borderwidth=0)
        self.box_listbox.pack(fill=tk.X, padx=12)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=12, pady=12)

        # Status / save
        self.status_label = tk.Label(right, text="Ready", bg="#16213e",
                                     fg="#68d391", font=("Consolas", 9),
                                     wraplength=200, justify=tk.LEFT)
        self.status_label.pack(padx=12, anchor="w")

        tk.Button(right, text="💾  Save  [S]",
                  command=self._save_current,
                  bg="#276749", fg="white", relief=tk.FLAT,
                  padx=8, pady=6, font=("Consolas", 10),
                  width=22).pack(padx=12, pady=(8, 0))

        tk.Label(right, text="\nKeyboard shortcuts:\n"
                              "← / →  prev / next\n"
                              "Del     delete selected\n"
                              "Esc     deselect\n"
                              "S       save\n"
                              "+ / −   zoom\n"
                              "Drag canvas  new box",
                 bg="#16213e", fg="#4a5568",
                 font=("Consolas", 9), justify=tk.LEFT).pack(
            padx=12, pady=8, anchor="w")

    # ── DATA LOADING ──────────────────────────────────────────────────────────

    def _load_split(self, split: str):
        img_dir = ROOT / "images" / split
        if not img_dir.exists():
            messagebox.showwarning("Not found",
                                   f"Folder not found:\n{img_dir}")
            return
        self.images = sorted(
            [p for p in img_dir.iterdir()
             if p.suffix.lower() in IMG_EXTS])
        self.img_idx = 0
        self._load_image()

    def _open_folder(self):
        folder = filedialog.askdirectory(title="Select images folder",
                                         initialdir=str(ROOT))
        if not folder:
            return
        folder = Path(folder)
        self.images = sorted(
            [p for p in folder.iterdir()
             if p.suffix.lower() in IMG_EXTS])
        self.img_idx = 0
        self._load_image()

    def _label_path_for(self, img_path: Path) -> Path:
        """Derive label path from image path."""
        parts = list(img_path.parts)
        try:
            idx = parts.index("images")
            parts[idx] = "labels"
        except ValueError:
            # Fallback: same folder, .txt extension
            return img_path.with_suffix(".txt")
        return Path(*parts).with_suffix(".txt")

    def _load_image(self):
        if not self.images:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300,
                text="No images found.", fill="#4a5568",
                font=("Consolas", 16))
            self.img_counter.config(text="0 / 0")
            return

        img_path = self.images[self.img_idx]
        self.pil_image = Image.open(img_path).convert("RGB")
        self.img_w, self.img_h = self.pil_image.size

        lbl_path = self._label_path_for(img_path)
        self.boxes = read_labels(lbl_path)

        self.selected_box = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.filename_label.config(text=img_path.name)
        self.img_counter.config(
            text=f"{self.img_idx + 1} / {len(self.images)}")
        self.root.title(f"YOLO BBox Editor — {img_path.name}")

        self._fit_image()
        self._redraw()
        self._update_box_list()
        self._update_sel_info()
        self._set_status(f"Loaded: {len(self.boxes)} box(es)")

    def _fit_image(self):
        """Set zoom so image fits the canvas."""
        if self.canvas_w <= 1 or self.img_w <= 1:
            return
        scale_w = self.canvas_w / self.img_w
        scale_h = self.canvas_h / self.img_h
        self.zoom = min(scale_w, scale_h) * 0.95
        # Center
        disp_w = self.img_w * self.zoom
        disp_h = self.img_h * self.zoom
        self.pan_x = (self.canvas_w - disp_w) / 2
        self.pan_y = (self.canvas_h - disp_h) / 2

    # ── DRAWING ───────────────────────────────────────────────────────────────

    def _redraw(self):
        self.canvas.delete("all")
        if self.pil_image is None:
            return

        # Draw image
        disp_w = max(1, int(self.img_w * self.zoom))
        disp_h = max(1, int(self.img_h * self.zoom))
        resized = self.pil_image.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.pan_x, self.pan_y,
                                  anchor=tk.NW, image=self.tk_image)

        # Draw boxes
        for i, box in enumerate(self.boxes):
            self._draw_box(i, box, selected=(i == self.selected_box))

        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom * 100)}%")

    def _img_to_canvas(self, px, py):
        return px * self.zoom + self.pan_x, py * self.zoom + self.pan_y

    def _canvas_to_img(self, cx, cy):
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def _draw_box(self, idx, box, selected=False):
        cls_id, cx, cy, w, h = box
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, self.img_w, self.img_h)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        cx2, cy2 = self._img_to_canvas(x2, y2)

        color = CLASS_COLORS.get(cls_id, "#ffffff")
        width = 3 if selected else 1.5
        dash   = () if selected else (4, 3)

        # Box rectangle
        self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=("#ff4444" if selected else color),
            width=width, dash=dash, tags=f"box_{idx}")

        # Class label background + text
        name = CLASS_NAMES.get(cls_id, f"cls {cls_id}")
        label = f" {cls_id}:{name[:12]} "
        lbl_y = max(cy1 - 18, 2)
        self.canvas.create_rectangle(
            cx1, lbl_y, cx1 + len(label) * 7, lbl_y + 16,
            fill=("#ff4444" if selected else color),
            outline="", tags=f"box_{idx}")
        self.canvas.create_text(
            cx1 + 4, lbl_y + 8,
            text=label.strip(), anchor=tk.W,
            fill="black", font=("Consolas", 9, "bold"),
            tags=f"box_{idx}")

        # Resize handles (only on selected)
        if selected:
            for hx, hy in [(cx1, cy1), (cx2, cy1),
                           (cx1, cy2), (cx2, cy2)]:
                s = HANDLE_SIZE // 2
                self.canvas.create_rectangle(
                    hx - s, hy - s, hx + s, hy + s,
                    fill="#ff4444", outline="white", width=1,
                    tags=f"handle_{idx}")

    # ── INTERACTION ───────────────────────────────────────────────────────────

    def _hit_test_handle(self, mx, my, box):
        """Return resize direction string if (mx,my) is near a corner."""
        cls_id, cx, cy, w, h = box
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, self.img_w, self.img_h)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        cx2, cy2 = self._img_to_canvas(x2, y2)
        tol = HANDLE_SIZE + 2
        corners = {
            "resize-NW": (cx1, cy1),
            "resize-NE": (cx2, cy1),
            "resize-SW": (cx1, cy2),
            "resize-SE": (cx2, cy2),
        }
        for mode, (hx, hy) in corners.items():
            if abs(mx - hx) <= tol and abs(my - hy) <= tol:
                return mode
        return None

    def _hit_test_box(self, mx, my):
        """Return index of topmost box under (mx,my), or None."""
        for i in reversed(range(len(self.boxes))):
            box = self.boxes[i]
            cls_id, cx, cy, w, h = box
            x1, y1, x2, y2 = yolo_to_pixel(
                cx, cy, w, h, self.img_w, self.img_h)
            cx1, cy1 = self._img_to_canvas(x1, y1)
            cx2, cy2 = self._img_to_canvas(x2, y2)
            if cx1 <= mx <= cx2 and cy1 <= my <= cy2:
                return i
        return None

    def _on_mouse_down(self, event):
        mx, my = event.x, event.y
        self.drag_start_x = mx
        self.drag_start_y = my

        # 1. Check if clicking a handle of the selected box
        if self.selected_box is not None:
            mode = self._hit_test_handle(mx, my,
                                         self.boxes[self.selected_box])
            if mode:
                self.drag_mode = mode
                self.drag_orig_box = list(self.boxes[self.selected_box])
                return

        # 2. Check if clicking any box
        hit = self._hit_test_box(mx, my)
        if hit is not None:
            self.selected_box = hit
            self.drag_mode = "move"
            self.drag_orig_box = list(self.boxes[hit])
            self._redraw()
            self._update_sel_info()
            self._update_box_list()
            return

        # 3. Start drawing a new box
        self.selected_box = None
        self.drag_mode = "draw"
        self._redraw()
        self._update_sel_info()

    def _on_mouse_drag(self, event):
        mx, my = event.x, event.y
        dx_c = mx - self.drag_start_x   # delta in canvas pixels
        dy_c = my - self.drag_start_y
        dx_i = dx_c / self.zoom          # delta in image pixels
        dy_i = dy_c / self.zoom

        if self.drag_mode == "draw":
            # Preview rectangle
            if self.draw_rect_id:
                self.canvas.delete(self.draw_rect_id)
            color = CLASS_COLORS.get(self.new_class_var.get(), "#ffffff")
            self.draw_rect_id = self.canvas.create_rectangle(
                self.drag_start_x, self.drag_start_y, mx, my,
                outline=color, width=2, dash=(4, 3))
            return

        if self.selected_box is None or self.drag_orig_box is None:
            return

        orig = self.drag_orig_box
        cls_id, cx0, cy0, w0, h0 = orig
        x1_0, y1_0, x2_0, y2_0 = yolo_to_pixel(
            cx0, cy0, w0, h0, self.img_w, self.img_h)

        if self.drag_mode == "move":
            nx1 = x1_0 + dx_i
            ny1 = y1_0 + dy_i
            nx2 = x2_0 + dx_i
            ny2 = y2_0 + dy_i
            # Clamp within image
            if nx1 < 0:
                nx2 -= nx1; nx1 = 0
            if ny1 < 0:
                ny2 -= ny1; ny1 = 0
            if nx2 > self.img_w:
                nx1 -= (nx2 - self.img_w); nx2 = self.img_w
            if ny2 > self.img_h:
                ny1 -= (ny2 - self.img_h); ny2 = self.img_h

        elif self.drag_mode == "resize-NW":
            nx1 = min(x1_0 + dx_i, x2_0 - MIN_BOX_PX)
            ny1 = min(y1_0 + dy_i, y2_0 - MIN_BOX_PX)
            nx2, ny2 = x2_0, y2_0

        elif self.drag_mode == "resize-NE":
            nx1, ny1 = x1_0, min(y1_0 + dy_i, y2_0 - MIN_BOX_PX)
            nx2 = max(x2_0 + dx_i, x1_0 + MIN_BOX_PX)
            ny2 = y2_0

        elif self.drag_mode == "resize-SW":
            nx1 = min(x1_0 + dx_i, x2_0 - MIN_BOX_PX)
            ny1 = y1_0
            nx2 = x2_0
            ny2 = max(y2_0 + dy_i, y1_0 + MIN_BOX_PX)

        elif self.drag_mode == "resize-SE":
            nx1, ny1 = x1_0, y1_0
            nx2 = max(x2_0 + dx_i, x1_0 + MIN_BOX_PX)
            ny2 = max(y2_0 + dy_i, y1_0 + MIN_BOX_PX)
        else:
            return

        ncx, ncy, nw, nh = pixel_to_yolo(nx1, ny1, nx2, ny2,
                                           self.img_w, self.img_h)
        self.boxes[self.selected_box] = [cls_id, ncx, ncy, nw, nh]
        self._redraw()

    def _on_mouse_up(self, event):
        mx, my = event.x, event.y

        if self.drag_mode == "draw":
            if self.draw_rect_id:
                self.canvas.delete(self.draw_rect_id)
                self.draw_rect_id = None

            # Convert canvas coords to image coords
            ix1, iy1 = self._canvas_to_img(
                min(self.drag_start_x, mx),
                min(self.drag_start_y, my))
            ix2, iy2 = self._canvas_to_img(
                max(self.drag_start_x, mx),
                max(self.drag_start_y, my))

            # Ignore tiny drags (click without drag)
            if abs(ix2 - ix1) < MIN_BOX_PX or abs(iy2 - iy1) < MIN_BOX_PX:
                self.drag_mode = None
                return

            # Clamp
            ix1 = max(0, min(ix1, self.img_w))
            iy1 = max(0, min(iy1, self.img_h))
            ix2 = max(0, min(ix2, self.img_w))
            iy2 = max(0, min(iy2, self.img_h))

            cls_id = self.new_class_var.get()
            cx, cy, w, h = pixel_to_yolo(ix1, iy1, ix2, iy2,
                                          self.img_w, self.img_h)
            self.boxes.append([cls_id, cx, cy, w, h])
            self.selected_box = len(self.boxes) - 1
            self._save_current(silent=True)
            self._redraw()
            self._update_box_list()
            self._update_sel_info()
            self._set_status(f"New box added (class {cls_id}). Saved.")

        elif self.drag_mode in ("move", "resize-NW", "resize-NE",
                                "resize-SW", "resize-SE"):
            self._save_current(silent=True)
            self._set_status("Saved.")

        self.drag_mode = None
        self.drag_orig_box = None

    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self._zoom(0.15, event.x, event.y)
        else:
            self._zoom(-0.15, event.x, event.y)

    def _zoom(self, delta, cx=None, cy=None):
        old_zoom = self.zoom
        self.zoom = max(0.1, min(10.0, self.zoom + delta))
        if cx is None:
            cx = self.canvas_w / 2
        if cy is None:
            cy = self.canvas_h / 2
        # Zoom towards cursor
        self.pan_x = cx - (cx - self.pan_x) * (self.zoom / old_zoom)
        self.pan_y = cy - (cy - self.pan_y) * (self.zoom / old_zoom)
        self._redraw()

    def _on_canvas_resize(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height
        if self.pil_image:
            self._fit_image()
            self._redraw()

    # ── ACTIONS ───────────────────────────────────────────────────────────────

    def delete_selected(self):
        if self.selected_box is None:
            return
        self.boxes.pop(self.selected_box)
        self.selected_box = None
        self._save_current(silent=True)
        self._redraw()
        self._update_box_list()
        self._update_sel_info()
        self._set_status("Box deleted. Saved.")

    def _deselect(self):
        self.selected_box = None
        self._redraw()
        self._update_sel_info()

    def _change_selected_class(self, event):
        if self.selected_box is None:
            return
        val = self.change_class_var.get()
        # val is the index in the combobox values list
        # values are "0: Longitudinal Crack", etc.
        try:
            cls_id = int(str(val).split(":")[0])
        except Exception:
            return
        self.boxes[self.selected_box][0] = cls_id
        self._save_current(silent=True)
        self._redraw()
        self._update_sel_info()
        self._set_status(f"Class changed to {cls_id}. Saved.")

    def _save_current(self, silent=False):
        if not self.images:
            return
        img_path = self.images[self.img_idx]
        lbl_path = self._label_path_for(img_path)
        write_labels(lbl_path, self.boxes)
        if not silent:
            self._set_status(f"Saved {len(self.boxes)} box(es) → {lbl_path.name}")

    def prev_image(self):
        if not self.images:
            return
        self._save_current(silent=True)
        self.img_idx = (self.img_idx - 1) % len(self.images)
        self._load_image()

    def next_image(self):
        if not self.images:
            return
        self._save_current(silent=True)
        self.img_idx = (self.img_idx + 1) % len(self.images)
        self._load_image()

    # ── UI UPDATES ────────────────────────────────────────────────────────────

    def _update_sel_info(self):
        if self.selected_box is None or self.selected_box >= len(self.boxes):
            self.sel_info.config(text="None")
            return
        box = self.boxes[self.selected_box]
        cls_id, cx, cy, w, h = box
        name = CLASS_NAMES.get(cls_id, f"?")
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, self.img_w, self.img_h)
        self.sel_info.config(
            text=f"Box #{self.selected_box}\n"
                 f"Class: {cls_id} {name}\n"
                 f"px: ({int(x1)},{int(y1)}) → ({int(x2)},{int(y2)})\n"
                 f"w={int(x2-x1)}px h={int(y2-y1)}px\n"
                 f"YOLO: cx={cx:.3f} cy={cy:.3f}\n"
                 f"      w={w:.3f} h={h:.3f}")

    def _update_box_list(self):
        self.box_listbox.delete(0, tk.END)
        for i, box in enumerate(self.boxes):
            cls_id = box[0]
            name = CLASS_NAMES.get(cls_id, f"?")
            marker = "▶ " if i == self.selected_box else "  "
            self.box_listbox.insert(tk.END, f"{marker}{i}: {cls_id}-{name[:10]}")
        if self.selected_box is not None:
            self.box_listbox.see(self.selected_box)
            self.box_listbox.selection_set(self.selected_box)

    def _on_listbox_select(self, event):
        sel = self.box_listbox.curselection()
        if sel:
            self.selected_box = sel[0]
            self._redraw()
            self._update_sel_info()

    def _set_status(self, msg: str):
        self.status_label.config(text=msg)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground="#0f0f1a",
                    background="#2d3561",
                    foreground="#e2e8f0",
                    selectbackground="#2d3561")

    app = BBoxEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
