# ─────────────────────────────────────────────────────────────────────────────
#  manual_segment.py  —  DHP Manual Annotation Tool
#
#  Interactive sky/vegetation annotation for fisheye DHP images.
#  Produces binary PNG masks (255 = sky, 0 = vegetation) at original resolution.
#
#  Usage:
#    python manual_segment.py --input ./images/ --output ./masks/
#    python manual_segment.py --input ./images/    (output defaults to ./manual_masks/)
#
#  Interaction:
#    S key          → SKY paint mode
#    V key          → VEGETATION paint mode
#    B key          → BOX selection mode
#    Click          → flood-fill from clicked pixel
#    Drag           → brush paint (fine detail)
#    [ / ]          → brush size ±3 px
#    Ctrl+Z         → undo
#    Enter          → save + advance to next image
#    ← / →          → prev / next image
#    Tab            → toggle overlay ↔ B&W view
#    Scroll / pinch → zoom
#    Space + drag   → pan     |   Space tap → fit to canvas
#    Esc            → clear box selection
#
#  Initialisation buttons:
#    Threshold buttons (Otsu / ISODATA / Li / Manual) — auto-initialise mask
#    Apply-to-box row — apply any source only within the drawn box region
#
#  Auto-save:
#    Drafts are written 2 s after each edit to manual_masks_draft/.
#    On reload the draft is restored automatically so work is never lost.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from skimage.filters import threshold_otsu, threshold_isodata, threshold_li
from scipy.ndimage import label as nd_label

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Interactive DHP sky/vegetation annotation tool.")
parser.add_argument("--input",  default=None,
                    help="Folder of images to annotate "
                         "(default: ./images or prompted at start)")
parser.add_argument("--output", default=None,
                    help="Folder to write binary masks "
                         "(default: ./manual_masks/)")
# Allow running without args when double-clicked
_args = parser.parse_args()

IMG_EXTS = (".jpg", ".JPG", ".jpeg", ".JPEG",
            ".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF")


def _find_default_input():
    """Look for an 'images' folder next to this script."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"),
        os.path.join(os.getcwd(), "images"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _list_images(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder)
                  if any(f.endswith(e) for e in IMG_EXTS))


# ── Resolve paths ─────────────────────────────────────────────────────────────
PHOTOS_DIR = _args.input
if PHOTOS_DIR is None:
    PHOTOS_DIR = _find_default_input()
if PHOTOS_DIR is None or not os.path.isdir(PHOTOS_DIR):
    # Defer the error to App startup so tkinter can show a file dialog
    PHOTOS_DIR = None

_ROOT   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = _args.output or os.path.join(_ROOT, "manual_masks")
DRAFT_DIR = os.path.join(OUT_DIR, "_drafts")

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(DRAFT_DIR, exist_ok=True)

CANVAS       = 680
MIN_Z, MAX_Z = 0.05, 30.0
DRAW_MS      = 16
PREV_SIZE    = 512
CONTROLS_H   = 310


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("DHP Segmentation")
        self.root.configure(bg="#1a1a1a")

        self.orig    = None
        self.mask    = None
        self.edges   = None
        self._ch_sm  = None
        self._ed_sm  = None
        self._sh = self._sw = 1
        self._preview = None

        self.fname   = ""
        self.idx     = 0
        self.zoom    = 1.0
        self.pan_x   = 0.0
        self.pan_y   = 0.0
        self.brush   = 12
        self.undos   = []
        self.last_xy = None
        self.dragging = False
        self._pstart  = None
        self.show_bw  = False
        self._pending = False
        self.mode     = "sky"
        self.space_held      = False
        self._space_dragged  = False
        self._hover_xy       = None
        self._dirty          = False
        self._draft_job      = None
        self._box            = None
        self._box_start      = None
        self._box_thresh_preview = None
        self._path_overrides = {}

        global CANVAS
        CANVAS = self._pick_canvas_size()

        # ── Resolve image folder ──────────────────────────────────────────────
        global PHOTOS_DIR
        if PHOTOS_DIR is None:
            from tkinter import filedialog, messagebox
            PHOTOS_DIR = filedialog.askdirectory(
                title="Select folder containing DHP images")
            if not PHOTOS_DIR:
                messagebox.showerror("No folder", "No image folder selected.")
                root.destroy()
                return

        self.filenames = _list_images(PHOTOS_DIR)
        if not self.filenames:
            from tkinter import messagebox
            messagebox.showerror("No images",
                                 f"No images found in:\n{PHOTOS_DIR}")
            root.destroy()
            return

        self.done = {os.path.splitext(f)[0]
                     for f in os.listdir(OUT_DIR)
                     if f.lower().endswith(".png") and not f.startswith("_")}

        self._build_ui()
        self._bind()
        done_indices = [i for i, f in enumerate(self.filenames)
                        if os.path.splitext(f)[0] in self.done]
        start = done_indices[-1] if done_indices else 0
        self.load(start)

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        G, G2 = "#1a1a1a", "#222"

        def lbl(p, t, fg="#ccc", **kw):
            kw.setdefault("font", ("Helvetica", 9))
            return tk.Label(p, text=t, bg=p["bg"], fg=fg, **kw)

        def mkbtn(p, t, cmd, bg="#d0d0d0", fg=None):
            try:
                r = int(bg[1:3], 16); g = int(bg[3:5], 16); b = int(bg[5:7], 16)
                text_col = "#ffffff" if 0.299*r + 0.587*g + 0.114*b < 140 else "#111111"
            except Exception:
                text_col = "#111111"
            if fg is not None:
                text_col = fg
            return tk.Button(p, text=t, command=cmd, bg=bg, fg=text_col,
                             relief=tk.RAISED, padx=8, pady=3,
                             font=("Helvetica", 9, "bold"), cursor="hand2",
                             activebackground=bg, activeforeground=text_col)

        # Title
        top = tk.Frame(self.root, bg=G)
        top.pack(fill=tk.X, padx=8, pady=(6, 2))
        self.lbl_title = lbl(top, "", fg="white", font=("Helvetica", 11, "bold"))
        self.lbl_title.pack(side=tk.LEFT)
        self.lbl_count = lbl(top, "", fg="#666")
        self.lbl_count.pack(side=tk.RIGHT)

        # Stats bar
        stats_row = tk.Frame(self.root, bg="#111122", pady=4)
        stats_row.pack(fill=tk.X, padx=8, pady=(0, 2))

        tk.Label(stats_row, text="SKY", bg="#111122", fg="#5aafff",
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.lbl_sky_pct = tk.Label(stats_row, text="—", bg="#111122", fg="#5aafff",
                                    font=("Helvetica", 18, "bold"), width=7, anchor="w")
        self.lbl_sky_pct.pack(side=tk.LEFT, padx=(0, 18))

        tk.Label(stats_row, text="VEG", bg="#111122", fg="#5dcc88",
                 font=("Helvetica", 13, "bold")).pack(side=tk.LEFT, padx=(0, 2))
        self.lbl_veg_pct = tk.Label(stats_row, text="—", bg="#111122", fg="#5dcc88",
                                    font=("Helvetica", 18, "bold"), width=7, anchor="w")
        self.lbl_veg_pct.pack(side=tk.LEFT, padx=(0, 18))

        # Go-to bar
        tk.Label(stats_row, text="Go to:", bg="#111122", fg="#aaa",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 4))
        self._goto_var = tk.StringVar()
        self._goto_entry = tk.Entry(stats_row, textvariable=self._goto_var,
                                    width=14, font=("Helvetica", 10),
                                    bg="#222244", fg="white",
                                    insertbackground="white", relief=tk.FLAT)
        self._goto_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._goto_entry.bind("<Return>",     lambda e: self._goto_image())
        self._goto_entry.bind("<KeyRelease>", self._goto_autocomplete)
        tk.Button(stats_row, text="→", command=self._goto_image,
                  bg="#4a90d9", fg="white", font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, padx=6, pady=1, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(0, 12))

        # Mode
        mode_row = tk.Frame(self.root, bg=G, pady=5)
        mode_row.pack(fill=tk.X, padx=8, pady=(2, 1))
        lbl(mode_row, "  Mode:", fg="#fff",
            font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))

        self.btn_sky = tk.Button(
            mode_row, text="☀  SKY  (S)", command=lambda: self._set_mode("sky"),
            bg="#4a90d9", fg="white", relief=tk.RAISED,
            padx=16, pady=5, font=("Helvetica", 11, "bold"), cursor="hand2")
        self.btn_sky.pack(side=tk.LEFT, padx=4)

        self.btn_veg = tk.Button(
            mode_row, text="🌿  VEG  (V)", command=lambda: self._set_mode("veg"),
            bg="#b0b0b0", fg="#333", relief=tk.RAISED,
            padx=16, pady=5, font=("Helvetica", 11, "bold"), cursor="hand2")
        self.btn_veg.pack(side=tk.LEFT, padx=4)

        self.btn_box = tk.Button(
            mode_row, text="☐  BOX  (B)", command=lambda: self._set_mode("box"),
            bg="#b0b0b0", fg="#333", relief=tk.RAISED,
            padx=16, pady=5, font=("Helvetica", 11, "bold"), cursor="hand2")
        self.btn_box.pack(side=tk.LEFT, padx=4)

        lbl(mode_row, "   click=flood   drag=brush   BOX: drag to select region",
            fg="#888").pack(side=tk.LEFT, padx=8)

        # Init mask row
        r1 = tk.Frame(self.root, bg=G2, pady=4)
        r1.pack(fill=tk.X, padx=8, pady=(2, 1))

        lbl(r1, "Threshold:", fg="#777").pack(side=tk.LEFT, padx=(4, 4))
        self.meth = tk.StringVar(value="Otsu")
        for m in ["Otsu", "ISODATA", "Li", "Manual"]:
            ttk.Radiobutton(r1, text=m, variable=self.meth,
                            value=m, command=self._apply_thresh
                            ).pack(side=tk.LEFT, padx=2)
        lbl(r1, "  Ch:").pack(side=tk.LEFT, padx=(6, 2))
        self.ch = tk.StringVar(value="Blue")
        for c in ["Blue", "Green", "Red", "Bright"]:
            ttk.Radiobutton(r1, text=c, variable=self.ch,
                            value=c, command=self._apply_thresh
                            ).pack(side=tk.LEFT, padx=2)
        self.tval = tk.DoubleVar(value=128)
        ttk.Scale(r1, from_=0, to=255, length=90,
                  variable=self.tval, command=self._slider_moved
                  ).pack(side=tk.LEFT, padx=5)
        self.lbl_tv = lbl(r1, "128", fg="#4a90d9")
        self.lbl_tv.pack(side=tk.LEFT)
        mkbtn(r1, "↺ Thresh", self._apply_thresh, "#888888").pack(side=tk.LEFT, padx=5)

        # Tool row
        r2 = tk.Frame(self.root, bg=G, pady=3)
        r2.pack(fill=tk.X, padx=8, pady=(1, 0))

        lbl(r2, "  Tolerance:").pack(side=tk.LEFT)
        self.tol = tk.IntVar(value=25)
        ttk.Scale(r2, from_=1, to=150, length=90, variable=self.tol,
                  command=lambda v: self.lbl_tol.config(text=str(int(float(v))))
                  ).pack(side=tk.LEFT, padx=3)
        self.lbl_tol = lbl(r2, "25", fg="#f0a040")
        self.lbl_tol.pack(side=tk.LEFT)

        lbl(r2, "   Edge barrier:").pack(side=tk.LEFT, padx=(8, 0))
        self.edge_sens = tk.IntVar(value=40)
        ttk.Scale(r2, from_=0, to=100, length=90, variable=self.edge_sens,
                  command=lambda v: self.lbl_edge.config(text=str(int(float(v))))
                  ).pack(side=tk.LEFT, padx=3)
        self.lbl_edge = lbl(r2, "40", fg="#a0d4a0")
        self.lbl_edge.pack(side=tk.LEFT)
        lbl(r2, "(0=off)", fg="#555").pack(side=tk.LEFT, padx=(2, 8))

        lbl(r2, "Brush:").pack(side=tk.LEFT)
        self.bsz = tk.IntVar(value=self.brush)
        ttk.Scale(r2, from_=1, to=120, length=80, variable=self.bsz,
                  command=lambda v: self._setbrush(int(float(v)))
                  ).pack(side=tk.LEFT, padx=3)
        self.lbl_b = lbl(r2, "12px", fg="#4a90d9")
        self.lbl_b.pack(side=tk.LEFT)

        for txt, cmd, bg in [
            ("💾 Save",  self.save_next,                "#3aaa3a"),
            ("Next →",   lambda: self.load(self.idx+1), "#d0d0d0"),
            ("← Prev",   lambda: self.load(self.idx-1), "#d0d0d0"),
            ("Undo ^Z",  self.undo,                     "#e08030"),
            ("Tab B&W",  self._toggle,                  "#9090cc"),
            ("📂 Open",  self._open_file,               "#cc88cc"),
        ]:
            mkbtn(r2, txt, cmd, bg).pack(side=tk.RIGHT, padx=2)

        # Apply-to-box row
        r3 = tk.Frame(self.root, bg="#1a1a2a", pady=3)
        r3.pack(fill=tk.X, padx=8, pady=(1, 0))
        lbl(r3, "  Apply to box:", fg="#aac4ff",
            font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))

        mkbtn(r3, "Threshold →", lambda: self._apply_to_box("thresh"),
              "#ccaa44").pack(side=tk.LEFT, padx=(6, 2))

        self.box_meth = tk.StringVar(value="Manual")
        for m in ["Otsu", "ISODATA", "Li", "Manual"]:
            ttk.Radiobutton(r3, text=m, variable=self.box_meth,
                            value=m, command=self._on_box_meth_switch
                            ).pack(side=tk.LEFT, padx=1)

        self.box_tval = tk.DoubleVar(value=128)
        self.box_tscale = ttk.Scale(r3, from_=0, to=255, length=100,
                                    variable=self.box_tval,
                                    command=self._on_box_tslider)
        self.box_tscale.pack(side=tk.LEFT, padx=4)
        self.lbl_btv = lbl(r3, "128", fg="#ffdd88")
        self.lbl_btv.pack(side=tk.LEFT)

        tk.Frame(r3, bg="#1a1a2a", width=10).pack(side=tk.LEFT)
        mkbtn(r3, "→ Sky", lambda: self._apply_to_box("sky"),  "#4a90d9").pack(side=tk.LEFT, padx=2)
        mkbtn(r3, "→ Veg", lambda: self._apply_to_box("veg"),  "#4ab87a").pack(side=tk.LEFT, padx=2)
        mkbtn(r3, "✗ Clear", self._clear_box, "#e05050").pack(side=tk.LEFT, padx=(8, 2))
        lbl(r3, "  (B key → drag to draw box)", fg="#555").pack(side=tk.LEFT)

        tk.Label(self.root,
                 text="  Scroll=zoom   Space+drag=pan   Space=fit   [ ]=brush   "
                      "Ctrl+Z=undo   Esc=clear box",
                 bg="#111", fg="#444", font=("Helvetica", 8)
                 ).pack(fill=tk.X, padx=8)

        self.cv = tk.Canvas(self.root, width=CANVAS, height=CANVAS,
                            bg="#111", highlightthickness=0, cursor="crosshair")
        self.cv.pack(padx=8, pady=(3, 8))

    def _pick_canvas_size(self):
        screen_h = self.root.winfo_screenheight()
        available = screen_h - 25 - 28 - CONTROLS_H - 24
        return max(400, min(900, available))

    def _set_mode(self, m):
        self.mode = m
        self._preview = None
        self.btn_sky.config(bg="#b0b0b0", fg="#333333")
        self.btn_veg.config(bg="#b0b0b0", fg="#333333")
        self.btn_box.config(bg="#b0b0b0", fg="#333333")
        if m == "sky":
            self.btn_sky.config(bg="#4a90d9", fg="#ffffff")
            self.cv.config(cursor="crosshair")
        elif m == "veg":
            self.btn_veg.config(bg="#3aaa3a", fg="#ffffff")
            self.cv.config(cursor="crosshair")
        else:
            self.btn_box.config(bg="#cc8800", fg="#ffffff")
            self.cv.config(cursor="tcross")
        self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Bindings
    # ─────────────────────────────────────────────────────────────────────────
    def _bind(self):
        c = self.cv
        c.bind("<ButtonPress-1>",   self._press)
        c.bind("<B1-Motion>",       self._drag)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<MouseWheel>",      self._wheel)
        c.bind("<Button-4>",        self._wheel)
        c.bind("<Button-5>",        self._wheel)
        c.bind("<Motion>",          self._hover)

        r = self.root
        r.bind("<KeyPress-s>",   lambda e: self._set_mode("sky"))
        r.bind("<KeyPress-S>",   lambda e: self._set_mode("sky"))
        r.bind("<KeyPress-v>",   lambda e: self._set_mode("veg"))
        r.bind("<KeyPress-V>",   lambda e: self._set_mode("veg"))
        r.bind("<KeyPress-b>",   lambda e: self._set_mode("box"))
        r.bind("<KeyPress-B>",   lambda e: self._set_mode("box"))
        r.bind("<Escape>",       lambda e: self._clear_box())
        r.bind("<Control-o>",    lambda e: self._open_file())
        r.bind("<Control-z>",    lambda e: self.undo())
        r.bind("<bracketleft>",  lambda e: self._setbrush(max(1, self.brush-3)))
        r.bind("<bracketright>", lambda e: self._setbrush(min(120, self.brush+3)))
        r.bind("<Return>",       lambda e: self.save_next())
        r.bind("<Right>",        lambda e: self.load(self.idx+1))
        r.bind("<Left>",         lambda e: self.load(self.idx-1))
        r.bind("<Tab>",          lambda e: self._toggle())
        r.bind("<KeyPress-space>",   self._space_dn)
        r.bind("<KeyRelease-space>", self._space_up)

    def _space_dn(self, e):
        self.space_held = True
        self._space_dragged = False
        self.cv.config(cursor="fleur")

    def _space_up(self, e):
        if self.space_held and not self._space_dragged:
            self.reset_zoom()
        self.space_held = False
        self._space_dragged = False
        self.cv.config(cursor="crosshair")

    # ─────────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────────
    def load(self, idx):
        if self._dirty:
            from tkinter import messagebox
            ans = messagebox.askyesnocancel(
                "Unsaved edits",
                f"'{self.fname}' has unsaved edits.\n\n"
                "Yes = Save now\nNo = Discard\nCancel = Stay")
            if ans is None:
                return
            if ans:
                self.save_next(); return
            else:
                self._dirty = False

        self.idx   = max(0, min(idx, len(self.filenames)-1))
        self.fname = self.filenames[self.idx]
        self.orig  = self.mask = self.edges = None
        self._ch_sm = self._ed_sm = self._preview = None
        try:
            full_path = self._path_overrides.get(
                self.fname, os.path.join(PHOTOS_DIR, self.fname))
            self.orig = np.array(Image.open(full_path).convert("RGB"))
        except Exception as e:
            print(f"[ERROR] Cannot load {self.fname}: {e}"); return
        self.undos     = []
        self._hover_xy = None
        self._dirty    = False
        if self._draft_job:
            self.root.after_cancel(self._draft_job)
            self._draft_job = None
        self._compute_edges()
        self._build_smalls()
        if not self._load_draft():
            self._apply_thresh()

    # ─────────────────────────────────────────────────────────────────────────
    # Edge map
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_edges(self):
        if self.orig is None: return
        gray = self.orig.mean(axis=2).astype(np.float32)
        gy   = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        gx   = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        mag  = gy + gx
        self.edges = (mag / (mag.max() + 1e-8) * 255).astype(np.float32)

    def _build_smalls(self):
        if self.orig is None: return
        h, w = self.orig.shape[:2]
        s    = PREV_SIZE
        self._sh, self._sw = h, w
        ys = np.clip(np.arange(s) * h // s, 0, h-1)
        xs = np.clip(np.arange(s) * w // s, 0, w-1)
        ch_full   = self._chanel()
        self._ch_sm = ch_full[np.ix_(ys, xs)]
        if self.edges is not None:
            self._ed_sm = self.edges[np.ix_(ys, xs)]

    # ─────────────────────────────────────────────────────────────────────────
    # Channel helper
    # ─────────────────────────────────────────────────────────────────────────
    def _chanel(self, arr=None):
        src = arr if arr is not None else self.orig
        if src is None: return None
        n = self.ch.get()
        if n == "Blue":   return src[:, :, 2].astype(np.float32)
        if n == "Green":  return src[:, :, 1].astype(np.float32)
        if n == "Red":    return src[:, :, 0].astype(np.float32)
        return src.astype(np.float32).mean(axis=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_thresh(self, *_):
        if self.orig is None: return
        ch = self._chanel()
        m  = self.meth.get()
        try:
            if   m == "Otsu":    t = float(threshold_otsu(ch))
            elif m == "ISODATA": t = float(threshold_isodata(ch))
            elif m == "Li":      t = float(threshold_li(ch))
            else:                t = float(self.tval.get())
        except Exception:
            t = float(self.tval.get())
        self.tval.set(t); self.lbl_tv.config(text=f"{t:.0f}")
        self.mask  = (ch > t).astype(np.uint8)
        self.undos = []
        self.reset_zoom()

    def _slider_moved(self, val):
        self.lbl_tv.config(text=f"{float(val):.0f}")
        if self.meth.get() == "Manual" and self.orig is not None:
            self.mask  = (self._chanel() > float(val)).astype(np.uint8)
            self.undos = []
            self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Draft auto-save
    # ─────────────────────────────────────────────────────────────────────────
    def _load_draft(self):
        if self.orig is None: return False
        stem  = os.path.splitext(self.fname)[0]
        path  = os.path.join(DRAFT_DIR, stem + ".png")
        if not os.path.exists(path): return False
        try:
            h, w  = self.orig.shape[:2]
            img   = Image.open(path).convert("L").resize((w, h), Image.NEAREST)
            self.mask  = (np.array(img) > 127).astype(np.uint8)
            self.undos = []
            self._dirty = True
            self.reset_zoom()
            return True
        except Exception:
            return False

    def _save_draft(self):
        self._draft_job = None
        if self.mask is None or self.fname == "": return
        stem = os.path.splitext(self.fname)[0]
        path = os.path.join(DRAFT_DIR, stem + ".png")
        try:
            Image.fromarray((self.mask*255).astype(np.uint8), mode="L").save(path)
        except Exception as e:
            print(f"[WARN] draft save failed: {e}")

    def _schedule_draft(self):
        if self._draft_job:
            self.root.after_cancel(self._draft_job)
        self._draft_job = self.root.after(2000, self._save_draft)
        self._dirty = True
        self._update_title_dirty()

    # ─────────────────────────────────────────────────────────────────────────
    # Box region tools
    # ─────────────────────────────────────────────────────────────────────────
    def _on_box_meth_switch(self, *_):
        self._box_meth_switched = True
        self._on_box_meth()

    def _on_box_meth(self, *_):
        bounds = self._box_bounds()
        m = self.box_meth.get()
        if bounds is None or self.orig is None:
            self._box_thresh_preview = None; return
        x0, y0, x1, y1 = bounds
        if m == "Manual":
            if getattr(self, "_box_meth_switched", False):
                self._box_meth_switched = False
                try:
                    t = float(threshold_otsu(self.orig[y0:y1, x0:x1, 2].astype(np.float32)))
                    self.box_tval.set(t); self.lbl_btv.config(text=f"{t:.0f}")
                except Exception: pass
            self._update_box_preview()
            self._schedule_draw(); return
        ch_crop = self._chanel()[y0:y1, x0:x1]
        try:
            if   m == "Otsu":    t = float(threshold_otsu(ch_crop))
            elif m == "ISODATA": t = float(threshold_isodata(ch_crop))
            elif m == "Li":      t = float(threshold_li(ch_crop))
            else: return
            self.box_tval.set(t); self.lbl_btv.config(text=f"{t:.0f}")
        except Exception: pass
        self._box_thresh_preview = None
        self._schedule_draw()

    def _on_box_tslider(self, val):
        self.lbl_btv.config(text=f"{float(val):.0f}")
        self.box_meth.set("Manual")
        self._update_box_preview()
        self._schedule_draw()

    def _update_box_preview(self):
        bounds = self._box_bounds()
        if bounds is None or self.orig is None:
            self._box_thresh_preview = None; return
        x0, y0, x1, y1 = bounds
        t = float(self.box_tval.get())
        sky_mask = self.orig[y0:y1, x0:x1, 2].astype(np.float32) > t
        self._box_thresh_preview = (sky_mask, x0, y0, x1, y1)

    def _clear_box(self):
        self._box = self._box_start = self._box_thresh_preview = None
        self._schedule_draw()

    def _box_bounds(self):
        if self._box is None or self.orig is None: return None
        h, w = self.orig.shape[:2]
        ix0, iy0, ix1, iy1 = self._box
        x0 = int(np.clip(min(ix0, ix1), 0, w-1))
        x1 = int(np.clip(max(ix0, ix1), 0, w))
        y0 = int(np.clip(min(iy0, iy1), 0, h-1))
        y1 = int(np.clip(max(iy0, iy1), 0, h))
        if x0 >= x1 or y0 >= y1: return None
        return x0, y0, x1, y1

    def _apply_to_box(self, source):
        bounds = self._box_bounds()
        if bounds is None:
            from tkinter import messagebox
            messagebox.showinfo("No box", "Draw a box first (B key, then drag).")
            return
        x0, y0, x1, y1 = bounds
        self._push_undo()
        if source == "sky":
            self.mask[y0:y1, x0:x1] = 1
        elif source == "veg":
            self.mask[y0:y1, x0:x1] = 0
        elif source == "thresh":
            m = self.box_meth.get()
            if m == "Manual":
                ch_crop = self.orig[y0:y1, x0:x1, 2].astype(np.float32)
                t = float(self.box_tval.get())
            else:
                ch_crop = self._chanel()[y0:y1, x0:x1]
                try:
                    if   m == "Otsu":    t = float(threshold_otsu(ch_crop))
                    elif m == "ISODATA": t = float(threshold_isodata(ch_crop))
                    elif m == "Li":      t = float(threshold_li(ch_crop))
                    else:                t = float(self.box_tval.get())
                except Exception:        t = float(self.box_tval.get())
            self.box_tval.set(t); self.lbl_btv.config(text=f"{t:.0f}")
            self.mask[y0:y1, x0:x1] = (ch_crop > t).astype(np.uint8)
        self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Flood fill
    # ─────────────────────────────────────────────────────────────────────────
    def _flood(self, ch, edges, iy, ix, tol, edge_sens):
        seed_val   = float(ch[iy, ix])
        candidates = (np.abs(ch.astype(np.float32) - seed_val) <= tol)
        if edge_sens > 0 and edges is not None:
            edge_thr = max(3.0, 255.0 - edge_sens * 2.52)
            barrier  = edges > edge_thr
            barrier  = (barrier
                        | np.roll(barrier,  1, axis=0) | np.roll(barrier, -1, axis=0)
                        | np.roll(barrier,  1, axis=1) | np.roll(barrier, -1, axis=1))
            candidates = candidates & ~barrier
        labeled, _ = nd_label(candidates)
        comp = int(labeled[iy, ix])
        if comp == 0:
            return np.zeros_like(candidates)
        return labeled == comp

    # ─────────────────────────────────────────────────────────────────────────
    # Hover preview
    # ─────────────────────────────────────────────────────────────────────────
    def _hover(self, e):
        if self.orig is None or self.dragging or self.space_held: return
        ix = int(np.clip(self._c2i(e.x, e.y)[0], 0, self.orig.shape[1]-1))
        iy = int(np.clip(self._c2i(e.x, e.y)[1], 0, self.orig.shape[0]-1))
        if (ix, iy) == self._hover_xy: return
        self._hover_xy = (ix, iy)
        if self._ch_sm is not None:
            s    = PREV_SIZE
            ix_s = int(np.clip(ix * s // self._sw, 0, s-1))
            iy_s = int(np.clip(iy * s // self._sh, 0, s-1))
            try:
                self._preview = self._flood(
                    self._ch_sm, self._ed_sm,
                    iy_s, ix_s, self.tol.get(), self.edge_sens.get())
            except Exception:
                self._preview = None
        self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Zoom / pan
    # ─────────────────────────────────────────────────────────────────────────
    def reset_zoom(self):
        if self.orig is None: return
        h, w = self.orig.shape[:2]
        self.zoom  = min(CANVAS/h, CANVAS/w)
        self.pan_x = 0.0; self.pan_y = 0.0
        self._schedule_draw()

    def _c2i(self, cx, cy):
        return self.pan_x + cx/self.zoom, self.pan_y + cy/self.zoom

    def _clamp(self):
        if self.orig is None: return
        h, w = self.orig.shape[:2]
        self.pan_x = max(0, min(self.pan_x, max(0, w - CANVAS/self.zoom)))
        self.pan_y = max(0, min(self.pan_y, max(0, h - CANVAS/self.zoom)))

    def _wheel(self, e):
        f = 1.2 if (e.num == 4 or e.delta > 0) else 1/1.2
        ix, iy = self._c2i(e.x, e.y)
        self.zoom  = max(MIN_Z, min(MAX_Z, self.zoom*f))
        self.pan_x = ix - e.x/self.zoom
        self.pan_y = iy - e.y/self.zoom
        self._clamp(); self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Draw
    # ─────────────────────────────────────────────────────────────────────────
    def _schedule_draw(self):
        if not self._pending:
            self._pending = True
            self.root.after(DRAW_MS, self._draw)

    def _draw(self):
        self._pending = False
        if self.orig is None or self.mask is None: return
        h, w = self.orig.shape[:2]
        vw = max(1, int(CANVAS/self.zoom)); vh = vw
        x0 = int(max(0, self.pan_x)); x1 = min(w, x0+vw)
        y0 = int(max(0, self.pan_y)); y1 = min(h, y0+vh)

        ci = Image.fromarray(self.orig[y0:y1, x0:x1]).resize((CANVAS, CANVAS), Image.NEAREST)
        cm = Image.fromarray(self.mask[y0:y1, x0:x1]*255, 'L').resize((CANVAS, CANVAS), Image.NEAREST)

        if self.show_bw:
            out = cm.convert("RGB")
        else:
            ia   = np.array(ci); ma = np.array(cm) == 255
            out_a = ia.copy()
            out_a[ ma] = (ia[ ma].astype(np.uint16)*45//100 + 141).clip(0, 255).astype(np.uint8)
            out_a[~ma] = (ia[~ma].astype(np.uint16)*70//100).clip(0, 255).astype(np.uint8)

            if self._preview is not None and not self.dragging:
                s    = PREV_SIZE
                px0  = max(0, int(x0*s//w)); px1 = min(s, int(x1*s//w)+1)
                py0  = max(0, int(y0*s//h)); py1 = min(s, int(y1*s//h)+1)
                crop = self._preview[py0:py1, px0:px1].astype(np.uint8)*255
                prev_disp = np.array(
                    Image.fromarray(crop, 'L').resize((CANVAS, CANVAS), Image.NEAREST)) > 127
                tint = (np.array([0, 210, 255], dtype=np.uint16) if self.mode == "sky"
                        else np.array([80, 255, 80], dtype=np.uint16))
                out_a[prev_disp] = (
                    out_a[prev_disp].astype(np.uint16)*35//100 + tint*65//100
                ).clip(0, 255).astype(np.uint8)
            out = Image.fromarray(out_a)

        self.tkimg = ImageTk.PhotoImage(out)
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor=tk.NW, image=self.tkimg)

        if self._box_thresh_preview is not None and self._box is not None:
            sky_mask, bx0, by0, bx1, by1 = self._box_thresh_preview
            cbx0 = (bx0 - self.pan_x) * self.zoom; cby0 = (by0 - self.pan_y) * self.zoom
            cbx1 = (bx1 - self.pan_x) * self.zoom; cby1 = (by1 - self.pan_y) * self.zoom
            bw = max(1, int(cbx1 - cbx0)); bh = max(1, int(cby1 - cby0))
            sky_img  = Image.fromarray(sky_mask.astype(np.uint8)*255, 'L').resize((bw, bh), Image.NEAREST)
            sky_arr  = np.array(sky_img) > 127
            prev_rgb = np.zeros((bh, bw, 3), dtype=np.uint8)
            prev_rgb[ sky_arr] = [100, 180, 255]
            prev_rgb[~sky_arr] = [80,  180,  80]
            prev_pil = Image.fromarray(prev_rgb, 'RGB').convert('RGBA')
            prev_arr = np.array(prev_pil); prev_arr[:, :, 3] = 120
            prev_pil = Image.fromarray(prev_arr, 'RGBA')
            self._box_prev_tk = ImageTk.PhotoImage(prev_pil)
            self.cv.create_image(int(cbx0), int(cby0), anchor=tk.NW, image=self._box_prev_tk)

        if self._box is not None:
            ix0, iy0, ix1, iy1 = self._box
            cx0 = (ix0 - self.pan_x)*self.zoom; cy0 = (iy0 - self.pan_y)*self.zoom
            cx1 = (ix1 - self.pan_x)*self.zoom; cy1 = (iy1 - self.pan_y)*self.zoom
            self.cv.create_rectangle(cx0, cy0, cx1, cy1, outline="#ffdd00", width=2, dash=(6, 4))
            self.cv.create_rectangle(cx0, cy0, cx1, cy1, outline="#ffaa00", width=1, dash=(1, 6))

        self._update_label()

    def _toggle(self):
        self.show_bw = not self.show_bw; self._schedule_draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Mouse / brush
    # ─────────────────────────────────────────────────────────────────────────
    def _flood_click(self, ix, iy, val):
        if self.orig is None: return
        ix = int(np.clip(ix, 0, self.orig.shape[1]-1))
        iy = int(np.clip(iy, 0, self.orig.shape[0]-1))
        region = self._flood(self._chanel(), self.edges, iy, ix,
                             self.tol.get(), self.edge_sens.get())
        self._push_undo()
        self.mask[region] = val

    def _paint(self, ix, iy, val):
        if self.mask is None: return
        h, w = self.mask.shape; r = self.brush
        x0 = max(0, int(ix-r)); x1 = min(w, int(ix+r)+1)
        y0 = max(0, int(iy-r)); y1 = min(h, int(iy+r)+1)
        if x0 >= x1 or y0 >= y1: return
        xs = np.arange(x0, x1); ys = np.arange(y0, y1)
        xx, yy = np.meshgrid(xs, ys)
        self.mask[y0:y1, x0:x1][(xx-ix)**2 + (yy-iy)**2 <= r*r] = val

    def _stroke(self, p0, p1, val):
        x0, y0 = p0; x1, y1 = p1
        d = max(abs(x1-x0), abs(y1-y0), 1)
        for i in range(max(1, int(d))+1):
            t = i / max(1, int(d))
            self._paint(x0+t*(x1-x0), y0+t*(y1-y0), val)

    def _press(self, e):
        self.dragging = False
        self._pstart  = (e.x, e.y, self.pan_x, self.pan_y)
        self.last_xy  = self._c2i(e.x, e.y)
        if self.mode == "box":
            self._box_start = self._c2i(e.x, e.y)

    def _drag(self, e):
        if self.space_held:
            self._space_dragged = True
            sx, sy, px, py = self._pstart
            self.pan_x = px - (e.x-sx)/self.zoom
            self.pan_y = py - (e.y-sy)/self.zoom
            self._clamp(); self._schedule_draw(); return
        if self.mode == "box":
            cur = self._c2i(e.x, e.y)
            if self._box_start:
                x0, y0 = self._box_start
                self._box = (x0, y0, cur[0], cur[1])
                self.dragging = True
                self._schedule_draw()
            return
        cur = self._c2i(e.x, e.y)
        if not self.dragging:
            self.dragging = True
            self._preview = None
            self._push_undo()
        val = 1 if self.mode == "sky" else 0
        self._stroke(self.last_xy, cur, val)
        self.last_xy = cur
        self._schedule_draw()

    def _release(self, e):
        if self.space_held:
            self.dragging = False; return
        if self.mode == "box":
            self.dragging = False
            self._box_start = None
            self._on_box_meth()
            self._schedule_draw(); return
        if not self.dragging and self.last_xy is not None:
            val = 1 if self.mode == "sky" else 0
            self._flood_click(*self.last_xy, val)
            self._schedule_draw()
        self.dragging = False
        self.last_xy  = None

    # ─────────────────────────────────────────────────────────────────────────
    # Undo / brush / labels / save
    # ─────────────────────────────────────────────────────────────────────────
    def _push_undo(self):
        self.undos.append(self.mask.copy())
        if len(self.undos) > 40: self.undos.pop(0)
        self._schedule_draft()

    def undo(self):
        if self.undos:
            self.mask = self.undos.pop(); self._schedule_draw()

    def _setbrush(self, v):
        self.brush = v; self.bsz.set(v); self.lbl_b.config(text=f"{v}px")

    def _update_title_dirty(self):
        if self.mask is None: return
        stem      = os.path.splitext(self.fname)[0]
        mk        = " ✓" if stem in self.done else ""
        dirty_mark = "  ● unsaved" if self._dirty else ""
        sky_pct   = self.mask.mean() * 100
        self.lbl_title.config(
            text=f"[{self.idx+1}/{len(self.filenames)}] {self.fname}{mk}  "
                 f"zoom={self.zoom:.1f}×{dirty_mark}")
        self.lbl_count.config(text=f"Saved {len(self.done)}/{len(self.filenames)}")
        self.lbl_sky_pct.config(text=f"{sky_pct:.1f}%")
        self.lbl_veg_pct.config(text=f"{100.0 - sky_pct:.1f}%")

    def _update_label(self):
        self._update_title_dirty()

    def _open_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Open image", initialdir=PHOTOS_DIR,
            filetypes=[("Image files", "*.jpg *.JPG *.jpeg *.png *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path: return
        fname = os.path.basename(path)
        for i, f in enumerate(self.filenames):
            if f == fname or os.path.join(PHOTOS_DIR, f) == path:
                self.load(i); return
        self.filenames.insert(self.idx + 1, fname)
        self._path_overrides[fname] = path
        self.load(self.idx + 1)

    def _goto_image(self):
        query = self._goto_var.get().strip().lower()
        if not query: return
        for i, f in enumerate(self.filenames):
            if query in f.lower() or query in os.path.splitext(f)[0].lower():
                self._goto_var.set("")
                self.load(i); return
        self._goto_entry.config(bg="#552222")
        self.root.after(600, lambda: self._goto_entry.config(bg="#222244"))

    def _goto_autocomplete(self, event=None):
        query = self._goto_var.get().strip().lower()
        if not query:
            self._goto_entry.config(bg="#222244"); return
        for f in self.filenames:
            if query in f.lower() or query in os.path.splitext(f)[0].lower():
                self._goto_entry.config(bg="#1a3322"); return
        self._goto_entry.config(bg="#332211")

    def save_next(self):
        stem = os.path.splitext(self.fname)[0]
        Image.fromarray((self.mask*255).astype(np.uint8), mode="L"
                        ).save(os.path.join(OUT_DIR, stem + ".png"))
        self.done.add(stem)
        self._dirty = False
        draft_path = os.path.join(DRAFT_DIR, stem + ".png")
        if os.path.exists(draft_path):
            try: os.remove(draft_path)
            except Exception: pass
        print(f"  ✓ [{self.idx+1}/{len(self.filenames)}] {self.fname}  "
              f"sky={self.mask.mean()*100:.1f}%")
        self.load(self.idx + 1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(True, True)
    App(root)
    root.mainloop()
