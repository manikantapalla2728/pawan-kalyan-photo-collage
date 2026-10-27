#!/usr/bin/env python3
"""
THE 1000-SCREEN MOSAIC
======================
A live, GPU-accelerated cinematic animation.  Over a thousand miniature
photographic screens drift in the dark, swarm, and then assemble into a
portrait that matches the reference photograph in assets/reference/.

    python main.py                 # live window, starts playing immediately
    python main.py --quality high  # more screens (low / medium / high)
    python main.py --help

Keys:  SPACE pause/resume   R restart   F fullscreen   ESC exit
       Q cycle quality      B toggle post-FX (bloom / depth of field)

Rendering: ModernGL (OpenGL 3.3) instanced boxes, one atlas of photo tiles,
HDR framebuffer, MSAA, temporal motion trails, depth of field, bloom,
tone-mapping, vignette and film grain.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PHOTO_DIR = ROOT / "assets" / "photos"
REF_DIR = ROOT / "assets" / "reference"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ----------------------------------------------------------------------------
# Timeline (seconds) - matches the five scenes of the brief
# ----------------------------------------------------------------------------
T_DARK_END = 3.0
T_SWARM_END = 7.0
T_BUILD_END = 13.0
T_REVEAL_END = 17.0
T_TITLE_END = 19.0

PORTRAIT_H = 20.0          # world height of the finished portrait
CELL_ASPECT = 1.25         # screen width / height of the smallest screen
GRID_COLS = 72             # base grid columns (smallest screen = 1 cell)
GAP = 0.02                 # fraction of a cell left as dark gap between screens
FOV = 35.0
MATCH_FINAL = 0.88         # how strongly screens lock to the reference colours
DETAIL_K = 0.50            # how much of each screen's own photo detail survives

QUALITY = {
    "low":    dict(count=1600, msaa=0, dust=500,  scale=0.75),
    "medium": dict(count=4000, msaa=4, dust=1400, scale=1.00),
    "high":   dict(count=7000, msaa=4, dust=2600, scale=1.00),
}
QUALITY_ORDER = ["low", "medium", "high"]


# ----------------------------------------------------------------------------
# Small math helpers
# ----------------------------------------------------------------------------
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_io_cubic(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < 0.5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)


def pchip(x, xs, ys):
    """Monotone cubic interpolation, zero slope at both ends (no overshoot)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    n = len(xs)
    m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    x = float(np.clip(x, xs[0], xs[-1]))
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, n - 2))
    t = (x - xs[i]) / h[i]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, np.float32)
    f = np.asarray(target, np.float32) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def quat_axis_angle(axis, ang):
    s = np.sin(ang * 0.5)[:, None]
    return np.concatenate([axis * s, np.cos(ang * 0.5)[:, None]], axis=1)


def quat_to_identity(q, w):
    """Normalised lerp from q towards the identity rotation (shortest path)."""
    q = q * np.where(q[:, 3:4] < 0, -1.0, 1.0)
    ident = np.zeros_like(q)
    ident[:, 3] = 1.0
    out = q * (1.0 - w)[:, None] + ident * w[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------
def imread_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def placeholder_photos(n=10, seed=3):
    """Procedural 'film stills' used only when assets/photos is empty."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h, w = 600, 800
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        c0, c1 = rng.random(3), rng.random(3)
        t = (yy / h * 0.6 + xx / w * 0.4)[..., None]
        img = c0 * (1 - t) + c1 * t
        for _ in range(8):
            cx, cy, r = rng.random() * w, rng.random() * h, 40 + rng.random() * 160
            d = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
            img = img * (1 - 0.6 * d) + rng.random(3) * 0.6 * d
        out.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return out


def load_photos():
    files = sorted(p for p in PHOTO_DIR.glob("*") if p.suffix.lower() in IMG_EXT) if PHOTO_DIR.exists() else []
    photos = [im for im in (imread_rgb(f) for f in files) if im is not None]
    if not photos:
        print("[mosaic] WARNING: no photographs found in assets/photos/ - using procedural placeholders.")
        print("         Drop your own .jpg/.png files there for the real thing.")
        photos = placeholder_photos()
    else:
        print(f"[mosaic] loaded {len(photos)} photograph(s) from assets/photos/")
    return photos


def load_target(photos):
    cands = sorted(p for p in REF_DIR.glob("target.*") if p.suffix.lower() in IMG_EXT) if REF_DIR.exists() else []
    img = imread_rgb(cands[0]) if cands else None
    if img is None:
        print("[mosaic] WARNING: assets/reference/target.png missing - using the first photograph as the target.")
        img = photos[0]
        h, w = img.shape[:2]
        cw = min(w, int(h * 0.8))
        x0 = (w - cw) // 2
        img = img[:, x0:x0 + cw]
    # keep working resolution sensible while preserving the aspect ratio
    h, w = img.shape[:2]
    s = 1024.0 / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    focus = None
    meta = REF_DIR / "target.json"
    if meta.exists():
        try:
            import json
            focus = json.loads(meta.read_text()).get("focus")
        except Exception as exc:  # noqa
            print("[mosaic] could not read target.json:", exc)
    if focus is None:
        focus = auto_focus(img)
    return img, [float(v) for v in focus]


def auto_focus(img):
    """Haar-cascade face detection with a sensible fallback."""
    try:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            H, W = gray.shape
            return [(x + w / 2) / W, (y + h / 2) / H, max(0.2, 0.9 * w / W), max(0.15, 0.95 * h / H)]
    except Exception:
        pass
    return [0.5, 0.55, 0.38, 0.30]


# ----------------------------------------------------------------------------
# Adaptive layout: a quadtree of screens, dense where the face has detail
# ----------------------------------------------------------------------------
def build_layout(target, count, focus, mask=None):
    th, tw = target.shape[:2]
    A = tw / th
    C = GRID_COLS
    R = int(4 * round(C * CELL_ASPECT / A / 4))
    fine = cv2.resize(target, (C * 4, R * 4), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(fine, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    grad = np.clip(grad / (np.percentile(grad, 99) + 1e-6), 0, 1.5)
    yy, xx = np.mgrid[0:R * 4, 0:C * 4]
    nx, ny = (xx + 0.5) / (C * 4), (yy + 0.5) / (R * 4)
    cx, cy, rx, ry = focus
    fw = np.exp(-(((nx - cx) / rx) ** 2 + ((ny - cy) / ry) ** 2))
    D = (0.12 + grad) * (0.30 + 1.70 * fw)
    if mask is not None:
        D = D * (1.0 + 0.9 * cv2.resize(mask, (C * 4, R * 4), interpolation=cv2.INTER_LINEAR))
    M1 = D.reshape(R, 4, C, 4).max(axis=(1, 3))
    M2 = M1.reshape(R // 2, 2, C // 2, 2).max(axis=(1, 3))
    M4 = M2.reshape(R // 4, 2, C // 4, 2).max(axis=(1, 3))

    def up(x):
        return np.repeat(np.repeat(x, 2, 0), 2, 1)

    def masks(T):
        s4 = M4 > T
        s2 = (M2 > T) & up(s4)
        return (~s4), (up(s4) & ~s2), up(s2)

    lo, hi = np.log(1e-3), np.log(5.0)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        l4, l2, l1 = masks(math.exp(mid))
        n = l4.sum() + l2.sum() + l1.sum()
        if n > count:
            lo = mid
        else:
            hi = mid
    l4, l2, l1 = masks(math.exp(hi))
    cells = []
    for s, m in ((4, l4), (2, l2), (1, l1)):
        ii, jj = np.nonzero(m)
        cells.append(np.stack([jj * s, ii * s, np.full_like(ii, s)], axis=1))
    cells = np.concatenate(cells, axis=0)
    return dict(cells=cells, C=C, R=R, A=A)


def person_mask(target, focus):
    """Rough foreground mask (GrabCut seeded by the face focus) used to
    order the reconstruction: head first, then body, background last."""
    h, w = target.shape[:2]
    sc = 300.0 / max(h, w)
    sw, sh = max(8, int(w * sc)), max(8, int(h * sc))
    small = cv2.cvtColor(cv2.resize(target, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    cx, cy, rx, ry = focus
    yy, xx = np.mgrid[0:sh, 0:sw]
    nx, ny = (xx + 0.5) / sw, (yy + 0.5) / sh
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    mask[(((nx - cx) / (rx * 1.9)) ** 2 + ((ny - cy) / (ry * 2.1)) ** 2) < 1] = cv2.GC_PR_FGD
    mask[(((nx - cx) / (rx * 0.9)) ** 2 + ((ny - cy) / (ry * 0.9)) ** 2) < 1] = cv2.GC_FGD
    b = max(2, int(0.03 * min(sw, sh)))
    mask[:b, :] = cv2.GC_BGD
    mask[:, :b] = cv2.GC_BGD
    mask[:, -b:] = cv2.GC_BGD
    try:
        bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
        cv2.grabCut(small, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    except Exception:
        fg = (((nx - cx) / (rx * 1.6)) ** 2 + ((ny - cy) / (ry * 2.0)) ** 2 < 1).astype(np.float32)
    return cv2.GaussianBlur(fg, (0, 0), 1.5)


# ----------------------------------------------------------------------------
# Photo tile atlas
# ----------------------------------------------------------------------------
def find_face(img):
    h, w = img.shape[:2]
    try:
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        s = 640.0 / max(h, w)
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (int(w * s), int(h * s)))
        faces = casc.detectMultiScale(small, 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return (x + fw / 2) / s, (y + fh / 2) / s, fw / s
    except Exception:
        pass
    return w * 0.5, h * 0.42, w * 0.30


GRADES = [
    lambda x: x,
    lambda x: x,
    lambda x: x * np.array([1.06, 1.0, 0.90]),                       # warm
    lambda x: x * np.array([0.92, 1.0, 1.08]),                       # cool
    lambda x: (x - 0.5) * 1.18 + 0.5,                                # punchy
    lambda x: x * 0.55 + x.mean(2, keepdims=True) * 0.45,            # muted
    lambda x: np.clip(x, 0, 1) ** 0.88 * np.array([1.0, 1.02, 0.96]),
]


def build_atlas(photos, aspect, seed=11, tile_w=128, gutter=4, atlas=2048):
    rng = np.random.default_rng(seed)
    th = int(round(tile_w / aspect))
    sw, sh = tile_w + 2 * gutter, th + 2 * gutter
    cols, rows = atlas // sw, atlas // sh
    K = cols * rows
    faces = [find_face(p) for p in photos]
    img = np.zeros((atlas, atlas, 3), np.uint8)
    rects = np.zeros((K, 4), np.float32)
    means = np.zeros((K, 3), np.float32)
    for k in range(K):
        pi = k % len(photos)
        p = photos[pi]
        h, w = p.shape[:2]
        fx, fy, fw = faces[pi]
        cw_max = min(w, h * aspect)
        r = rng.random()
        if r < 0.50:
            cw = fw * rng.choice([0.9, 1.2, 1.6, 2.1]) * rng.uniform(0.85, 1.15)
            cx = fx + rng.uniform(-0.45, 0.45) * fw
            cy = fy + rng.uniform(-0.40, 0.55) * fw
        elif r < 0.90:
            cw = cw_max * rng.uniform(0.07, 0.30)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        else:
            cw = cw_max * rng.uniform(0.45, 0.9)
            cx, cy = w * 0.5 + rng.uniform(-0.1, 0.1) * w, h * 0.5
        cw = float(np.clip(cw, 40, cw_max))
        ch = cw / aspect
        x0 = float(np.clip(cx - cw / 2, 0, w - cw))
        y0 = float(np.clip(cy - ch / 2, 0, h - ch))
        crop = p[int(y0):int(y0 + ch) + 1, int(x0):int(x0 + cw) + 1]
        tile = cv2.resize(crop, (tile_w, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if rng.random() < 0.5:
            tile = tile[:, ::-1]
        tile = np.clip(GRADES[rng.integers(len(GRADES))](tile), 0, 1)
        tile8 = (tile * 255 + 0.5).astype(np.uint8)
        padded = cv2.copyMakeBorder(tile8, gutter, gutter, gutter, gutter, cv2.BORDER_REPLICATE)
        c, rr = k % cols, k // cols
        img[rr * sh:(rr + 1) * sh, c * sw:(c + 1) * sw] = padded
        rects[k] = [(c * sw + gutter) / atlas, (rr * sh + gutter) / atlas,
                    (c * sw + gutter + tile_w) / atlas, (rr * sh + gutter + th) / atlas]
        means[k] = tile.reshape(-1, 3).mean(0)
    return img, rects, means


# ----------------------------------------------------------------------------
# The scene: per-screen kinematics (all vectorised with NumPy)
# ----------------------------------------------------------------------------
class Scene:
    def __init__(self, assets, count, seed=7):
        rng = np.random.default_rng(seed)
        target, focus = assets["target"], assets["focus"]
        pmask = person_mask(target, focus)
        lay = build_layout(target, count, focus, pmask)
        cells, C, R = lay["cells"], lay["C"], lay["R"]
        n = len(cells)
        self.n = n
        self.C, self.R = C, R
        A = lay["A"]
        self.Hp = PORTRAIT_H
        self.Wp = PORTRAIT_H * A
        self.cell_aspect = (self.Wp / C) / (self.Hp / R)

        s = cells[:, 2].astype(np.float32)
        cu = (cells[:, 0] + s / 2) / C
        cv_ = (cells[:, 1] + s / 2) / R
        self.tgt = np.stack([(cu - 0.5) * self.Wp, (0.5 - cv_) * self.Hp, np.zeros(n)], axis=1).astype(np.float32)
        self.cw = (self.Wp / C * s).astype(np.float32)          # final cell width in world units
        self.tgt_rect = np.stack([cells[:, 0] / C, cells[:, 1] / R,
                                  (cells[:, 0] + s) / C, (cells[:, 1] + s) / R], axis=1).astype(np.float32)

        # mean target colour per cell -> choose tiles
        small = cv2.resize(target, (C, R), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        cell_mean = np.empty((n, 3), np.float32)
        for i, (x0, y0, ss) in enumerate(cells):
            cell_mean[i] = small[y0:y0 + ss, x0:x0 + ss].reshape(-1, 3).mean(0)
        K = len(assets["atlas_means"])
        usage = np.zeros(K, np.float32)
        tile_idx = np.zeros(n, np.int32)
        for i in rng.permutation(n):
            d = ((assets["atlas_means"] - cell_mean[i]) ** 2).sum(1)
            cost = d + usage * (0.004 + 0.0012 * cells[i, 2]) + rng.random(K) * 0.01
            k = int(np.argmin(cost))
            tile_idx[i] = k
            usage[k] += 1
        self.tile_rect = assets["atlas_rects"][tile_idx]
        self.tile_mean = assets["atlas_means"][tile_idx]

        # ordering of the reconstruction: head/hair first -> clothing -> background
        mask = pmask
        mh, mw = mask.shape
        mi = np.clip((cv_ * mh).astype(int), 0, mh - 1)
        mj = np.clip((cu * mw).astype(int), 0, mw - 1)
        is_person = mask[mi, mj] > 0.5
        key = np.where(is_person, 0.02 + 0.70 * cv_, 0.80 + 0.20 * rng.random(n))
        key = key + rng.uniform(-0.025, 0.025, n)
        key = (key - key.min()) / (key.max() - key.min())
        self.t_start = (T_SWARM_END + 0.15 + key * 3.1 + rng.uniform(0, 0.25, n)).astype(np.float32)
        self.dur = rng.uniform(2.0, 2.55, n).astype(np.float32)
        self.t_end = self.t_start + self.dur
        self.bulge = rng.uniform(0.10, 0.26, n).astype(np.float32) * rng.choice([-1, 1], n)
        rv = rng.normal(size=(n, 3))
        self.rv = (rv / np.linalg.norm(rv, axis=1, keepdims=True)).astype(np.float32)

        # --- scattered phase -------------------------------------------------
        early = np.zeros(n, bool)
        early[rng.choice(n, min(n, 70), replace=False)] = True
        self.appear = np.where(early, rng.uniform(0.25, 2.9, n), 3.0 + 4.0 * rng.random(n) ** 1.25).astype(np.float32)
        base = np.stack([rng.uniform(-30, 30, n), rng.uniform(-17, 17, n), rng.uniform(-20, 22, n)], axis=1)
        # the early screens sit in front of the opening camera so the first seconds are not empty
        ne = int(early.sum())
        z = rng.uniform(4, 26, ne)
        half_h = (36 - z) * math.tan(math.radians(FOV / 2)) * 0.85
        base[early] = np.stack([rng.uniform(-1, 1, ne) * half_h * 1.6, rng.uniform(-1, 1, ne) * half_h, z], axis=1)
        self.sc_base = base.astype(np.float32)
        vdir = rng.normal(size=(n, 3))
        vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
        self.sc_vel = (vdir * rng.uniform(0.25, 0.85, (n, 1))).astype(np.float32)
        self.sc_amp = rng.uniform(0.4, 1.8, (n, 3)).astype(np.float32)
        self.sc_freq = rng.uniform(0.15, 0.45, (n, 3)).astype(np.float32)
        self.sc_ph = rng.uniform(0, 6.283, (n, 3)).astype(np.float32)
        self.sc_w = rng.uniform(0.55, 1.7, n).astype(np.float32)   # scattered width
        ax = rng.normal(size=(n, 3))
        self.axis = (ax / np.linalg.norm(ax, axis=1, keepdims=True)).astype(np.float32)
        self.th0 = rng.uniform(0, 6.283, n).astype(np.float32)
        self.om = (rng.uniform(0.12, 0.5, n) * rng.choice([-1, 1], n)).astype(np.float32)

        # --- swarm (a slowly rotating vortex around the portrait) -------------
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.sw_dir = d.astype(np.float32)
        self.sw_R = rng.uniform(13, 34, n).astype(np.float32)
        self.sw_ang0 = rng.uniform(0, 6.283, n).astype(np.float32)
        self.sw_om = rng.uniform(0.32, 0.62, n).astype(np.float32)
        self.d1 = rng.uniform(0, 1.2, n).astype(np.float32)
        self.seed = rng.random(n).astype(np.float32)

    # ------------------------------------------------------------------
    def update(self, t):
        """Return the dynamic per-instance array (N, 14):
        pos3, quat4, half3, state4(match, gold, bright, seed)."""
        n = self.n
        vis = smooth((t - self.appear) / 0.9)

        sp = self.sc_base + self.sc_vel * t + self.sc_amp * np.sin(self.sc_freq * t + self.sc_ph)

        rs = 1.0 - 0.28 * smooth((t - 3.0) / 4.5)
        ang = self.sw_ang0 + self.sw_om * t
        ca, sa = np.cos(ang), np.sin(ang)
        bx = self.sw_dir[:, 0] * self.sw_R * rs
        by = self.sw_dir[:, 1] * self.sw_R * rs * 0.42
        bz = self.sw_dir[:, 2] * self.sw_R * rs * 0.55
        x = bx * ca + bz * sa
        z = -bx * sa + bz * ca
        swarm = np.stack([x, by + 0.18 * x, z], axis=1)

        s1 = smooth((t - 3.0 - self.d1) / 3.0)[:, None]
        pre = sp + (swarm - sp) * s1

        u = np.clip((t - self.t_start) / self.dur, 0, 1)
        e = ease_io_cubic(u)
        dvec = self.tgt - pre
        dist = np.linalg.norm(dvec, axis=1)
        perp = np.cross(dvec, self.rv)
        perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-6
        bump = perp * (dist * self.bulge * np.sin(np.pi * u))[:, None]
        pos = pre + dvec * e[:, None] + bump

        fw = self.sc_w * (1.0 - 0.55 * s1[:, 0]) * (1 - e) + self.cw * (1 - GAP) * e
        hw = 0.5 * fw * vis
        hh = hw / self.cell_aspect
        hd = hh * (0.07 * (1 - e) + 0.06 * e) + 0.003

        th = self.th0 + self.om * t
        qs = quat_axis_angle(self.axis, th)
        w = 0.30 * s1[:, 0] * (1 - e) + e
        quat = quat_to_identity(qs, w)

        match = smooth((e - 0.30) / 0.70) * MATCH_FINAL
        tau = t - self.t_end
        land = np.where(tau > 0, np.exp(-np.clip(tau, 0, 20) * 2.1) * 1.5, 0.0)
        travel = 0.22 * np.sin(np.pi * u)
        intro = 0.10 * (1 - smooth((t - 3.0) / 3.0))
        gold = land + travel + intro
        bright = vis * (0.88 * (1 - e) + 1.0 * e)

        out = np.empty((n, 14), np.float32)
        out[:, 0:3] = pos
        out[:, 3:7] = quat
        out[:, 7] = hw
        out[:, 8] = hh
        out[:, 9] = hd
        out[:, 10] = match
        out[:, 11] = gold
        out[:, 12] = bright
        out[:, 13] = self.seed
        return out


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
class CameraRig:
    def __init__(self, scene):
        self.Wp, self.Hp = scene.Wp, scene.Hp

    def final_distance(self, aspect):
        th = math.tan(math.radians(FOV / 2))
        d_h = self.Hp / (0.84 * 2 * th)
        d_w = self.Wp / (0.86 * 2 * th * aspect)
        return max(d_h, d_w)

    def pose(self, t, aspect):
        dfin = self.final_distance(aspect)
        f = pchip(t, [0, 3, 7, 10.5, 13, 17, 19, 40], [0.92, 1.02, 1.00, 0.68, 0.57, 1.0, 1.035, 1.04])
        dist = f * dfin
        az = pchip(t, [0, 3, 7, 11, 13, 16, 19, 40], [-13, 6, 12, 5, 1.6, 0.4, 0.0, 0.0])
        az += 0.55 * math.sin(0.33 * t) * float(smooth((t - 15.0) / 2.0))
        el = pchip(t, [0, 3, 7, 11, 13, 17, 40], [3.5, 1.0, 3.0, 1.5, 0.0, 0.0, 0.0])
        roll = pchip(t, [0, 3, 8, 40], [4.0, 1.5, 0.0, 0.0])
        look_y = pchip(t, [0, 7, 11, 13, 17, 40], [0.0, 0.0, -0.5, -0.5, -1.15, -1.15])
        look_x = pchip(t, [0, 7, 40], [0.0, 0.0, 0.0])
        a, e = math.radians(az), math.radians(el)
        look = np.array([look_x, look_y, 0.0], np.float32)
        eye = look + dist * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)], np.float32)
        r = math.radians(roll)
        up = np.array([math.sin(r), math.cos(r), 0.0], np.float32)
        return eye, look, up, dist


# ----------------------------------------------------------------------------
# Title texture
# ----------------------------------------------------------------------------
def make_title_texture(text="PAWAN KALYAN"):
    import pygame
    pygame.font.init()
    try:
        font = pygame.font.SysFont("georgia,garamond,timesnewroman,palatino,dejavuserif,arial", 132)
    except Exception:
        font = pygame.font.Font(None, 132)
    glyphs = []
    for ch in text:
        if ch == " ":
            glyphs.append(None)
            continue
        s = font.render(ch, True, (255, 255, 255))
        a = np.frombuffer(pygame.image.tostring(s, "RGBA"), np.uint8).reshape(s.get_height(), s.get_width(), 4)[..., 3]
        glyphs.append(a.astype(np.float32) / 255.0)
    spacing, space_w, pad = 34, 60, 90
    gh = max(g.shape[0] for g in glyphs if g is not None)
    width = sum((g.shape[1] if g is not None else space_w) for g in glyphs) + spacing * (len(glyphs) - 1) + 2 * pad
    mask = np.zeros((gh + 2 * pad, width), np.float32)
    x = pad
    for g in glyphs:
        if g is None:
            x += space_w + spacing
            continue
        mask[pad:pad + g.shape[0], x:x + g.shape[1]] = np.maximum(mask[pad:pad + g.shape[0], x:x + g.shape[1]], g)
        x += g.shape[1] + spacing
    glow = cv2.GaussianBlur(mask, (0, 0), 22)
    glow = np.clip(glow * 2.2, 0, 1)
    text_col = np.array([1.0, 0.95, 0.84], np.float32)
    gold = np.array([1.0, 0.72, 0.30], np.float32)
    rgb = text_col * mask[..., None] + gold * (glow * (1 - mask))[..., None]
    alpha = np.clip(mask + glow * 0.55, 0, 1)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# GLSL
# ----------------------------------------------------------------------------
SCREEN_VS = r'''
#version 330
uniform mat4 u_viewproj;
in vec3 in_pos; in vec3 in_normal; in vec2 in_uv; in float in_kind;
in vec3 i_pos; in vec4 i_quat; in vec3 i_half; in vec4 i_state;
in vec4 i_tile; in vec3 i_mean; in vec4 i_tgt;
out vec3 v_wp; out vec3 v_n; out vec2 v_uv; out float v_kind;
flat out vec4 f_tile; flat out vec3 f_mean; flat out vec4 f_tgt; flat out vec4 f_state;
vec3 qrot(vec4 q, vec3 v){ return v + 2.0*cross(q.xyz, cross(q.xyz, v) + q.w*v); }
void main(){
    vec3 wp = i_pos + qrot(i_quat, in_pos * i_half);
    v_wp = wp;
    v_n = qrot(i_quat, in_normal);
    v_uv = in_uv; v_kind = in_kind;
    f_tile = i_tile; f_mean = i_mean; f_tgt = i_tgt; f_state = i_state;
    gl_Position = u_viewproj * vec4(wp, 1.0);
}
'''

SCREEN_FS = r'''
#version 330
uniform sampler2D u_atlas;
uniform sampler2D u_target;
uniform vec2  u_target_size;
uniform vec3  u_cam;
uniform mat4  u_view;
uniform float u_time, u_fog, u_aperture, u_focus, u_k, u_gloss, u_sweep, u_sweep_amt, u_edge;
uniform vec3  u_lpos[3];
uniform vec3  u_lcol[3];
in vec3 v_wp; in vec3 v_n; in vec2 v_uv; in float v_kind;
flat in vec4 f_tile; flat in vec3 f_mean; flat in vec4 f_tgt; flat in vec4 f_state;
out vec4 frag;
const vec3 GOLD = vec3(1.0, 0.74, 0.32);
const vec3 ICE  = vec3(0.45, 0.68, 1.0);
void main(){
    vec3 N = normalize(v_n);
    vec3 V = normalize(u_cam - v_wp);
    if (!gl_FrontFacing) N = -N;
    float dist  = length(u_cam - v_wp);
    float depth = -(u_view * vec4(v_wp, 1.0)).z;
    float coc = clamp(abs(depth - u_focus) / max(u_focus, 1.0) * 2.6 * u_aperture - 0.07, 0.0, 1.0);
    float match = f_state.x, gold = f_state.y, bright = f_state.z;
    vec3 col;
    if (v_kind < 0.5) {
        const float bz = 0.018;
        vec2 uv = v_uv;
        vec2 ed = min(uv, 1.0 - uv);
        float inside = smoothstep(bz - 0.004, bz + 0.004, min(ed.x, ed.y));
        vec2 duv = clamp((uv - bz) / (1.0 - 2.0 * bz), 0.0, 1.0);
        vec3 photo = texture(u_atlas, mix(f_tile.xy, f_tile.zw, duv), coc * 3.0).rgb;
        float cellpx = max((f_tgt.z - f_tgt.x) * u_target_size.x, 1.0);
        float lod = max(log2(cellpx) - 2.3, 0.0) + coc * 2.0;
        vec3 low = textureLod(u_target, mix(f_tgt.xy, f_tgt.zw, duv), lod).rgb;
        float kk = u_k * (0.50 + 0.50 * clamp(log2(cellpx / 8.0) / 2.0, 0.0, 1.0));
        vec3 matched = clamp(low + (photo - f_mean) * kk, 0.0, 1.0);
        vec3 disp = mix(photo, matched, match);
        vec3 lin = pow(disp, vec3(2.2)) * bright;
        float vig = 1.0 - 0.16 * pow(length(duv - 0.5) * 1.414, 2.4);
        lin *= vig;
        vec3 refl = vec3(0.0);
        for (int i = 0; i < 3; i++) {
            vec3 L = normalize(u_lpos[i] - v_wp);
            vec3 R = reflect(-L, N);
            refl += u_lcol[i] * pow(max(dot(R, V), 0.0), 70.0);
        }
        float fres = pow(1.0 - max(dot(N, V), 0.0), 4.0);
        lin += refl * 0.42 * u_gloss * bright;
        lin += mix(ICE, GOLD, 0.5 + 0.5 * N.x) * fres * 0.22 * bright;
        float sw = exp(-pow((dot(v_wp.xy, vec2(0.8, 0.6)) - u_sweep) * 0.17, 2.0)) * u_sweep_amt;
        lin += lin * sw * 0.9 + GOLD * sw * 0.07;
        vec3 bezel = vec3(0.012, 0.013, 0.017) * bright + refl * 0.5 * u_gloss + GOLD * gold * 0.9
                   + ICE * fres * 0.35 * u_edge * bright;
        col = mix(bezel, lin, inside);
    } else {
        vec3 c = vec3(0.004, 0.005, 0.007);
        for (int i = 0; i < 3; i++) {
            vec3 L = normalize(u_lpos[i] - v_wp);
            vec3 R = reflect(-L, N);
            c += u_lcol[i] * (max(dot(N, L), 0.0) * 0.035 + pow(max(dot(R, V), 0.0), 18.0) * 0.55 * u_edge);
        }
        float rim = pow(1.0 - abs(dot(N, V)), 2.0);
        c += mix(ICE, vec3(1.0), 0.35) * rim * 0.30 * u_edge * bright;
        c += GOLD * gold * 0.75;
        col = c * (0.35 + 0.65 * bright);
    }
    col *= exp(-dist * u_fog);
    frag = vec4(col, coc);
}
'''

DUST_VS = r'''
#version 330
uniform mat4 u_viewproj; uniform mat4 u_view;
uniform float u_time, u_pxscale, u_focus, u_alpha;
in vec3 in_base; in vec4 in_seed;
out vec3 v_col; out float v_a;
void main(){
    vec3 vel = (in_seed.xyz * 2.0 - 1.0) * vec3(0.55, 0.35, 0.55);
    vec3 vol = vec3(80.0, 44.0, 80.0);
    vec3 p = mod(in_base + vel * u_time + vol * 0.5, vol) - vol * 0.5;
    vec4 vp = u_view * vec4(p, 1.0);
    float depth = max(-vp.z, 0.5);
    float coc = clamp(abs(depth - u_focus) / max(u_focus, 1.0) * 2.0, 0.0, 1.0);
    float rw = 0.05 + 0.20 * in_seed.y * in_seed.y;
    float size = rw * u_pxscale / depth * (1.0 + coc * 2.2);
    gl_PointSize = clamp(size, 1.5, 70.0);
    float tw = 0.6 + 0.4 * sin(u_time * (0.8 + in_seed.w * 2.0) + in_seed.w * 40.0);
    v_a = u_alpha * tw * smoothstep(1.5, 7.0, depth) * (1.0 - smoothstep(55.0, 95.0, depth))
          / pow(1.0 + coc * 2.2, 2.0) * (0.35 + 0.65 * in_seed.y);
    v_col = mix(vec3(0.55, 0.75, 1.0), vec3(1.0, 0.76, 0.38), in_seed.z);
    gl_Position = u_viewproj * vec4(p, 1.0);
}
'''
DUST_FS = r'''
#version 330
in vec3 v_col; in float v_a;
out vec4 frag;
void main(){
    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float r = dot(pc, pc);
    if (r > 1.0) discard;
    frag = vec4(v_col * exp(-r * 3.6) * v_a, 0.0);
}
'''

HALO_VS = r'''
#version 330
uniform mat4 u_viewproj; uniform vec2 u_ext;
in vec2 in_p; out vec2 v_p;
void main(){ v_p = in_p * u_ext; gl_Position = u_viewproj * vec4(v_p, -1.4, 1.0); }
'''
HALO_FS = r'''
#version 330
uniform vec2 u_half; uniform float u_amt, u_time;
in vec2 v_p; out vec4 frag;
void main(){
    vec2 q = abs(v_p) - u_half;
    float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
    float s = max(sd, 0.0);
    float g = exp(-s * 0.55) * 0.60 + exp(-s * 0.20) * 0.10;
    float pulse = 0.93 + 0.07 * sin(u_time * 1.6);
    frag = vec4(vec3(1.0, 0.70, 0.28) * g * u_amt * pulse, 0.0);
}
'''

FS_VS = r'''
#version 330
out vec2 v_uv;
void main(){
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
'''
ACCUM_FS = r'''
#version 330
uniform sampler2D u_cur; uniform sampler2D u_prev; uniform float u_decay;
in vec2 v_uv; out vec4 f;
void main(){
    vec4 c = texture(u_cur, v_uv);
    vec4 p = texture(u_prev, v_uv);
    f = vec4(mix(c.rgb, p.rgb, u_decay), c.a);
}
'''
DOWN_FS = r'''
#version 330
uniform sampler2D u_src; uniform vec2 u_texel; uniform float u_thr; uniform float u_use_thr;
in vec2 v_uv; out vec4 f;
void main(){
    vec4 c = texture(u_src, v_uv + u_texel * vec2(-0.5, -0.5))
           + texture(u_src, v_uv + u_texel * vec2( 0.5, -0.5))
           + texture(u_src, v_uv + u_texel * vec2(-0.5,  0.5))
           + texture(u_src, v_uv + u_texel * vec2( 0.5,  0.5));
    c *= 0.25;
    if (u_use_thr > 0.5) {
        float br = max(max(c.r, c.g), c.b);
        float w = smoothstep(u_thr, u_thr + 0.55, br);
        c.rgb *= w;
    }
    f = c;
}
'''
BLUR_FS = r'''
#version 330
uniform sampler2D u_src; uniform vec2 u_dir;
in vec2 v_uv; out vec4 f;
void main(){
    vec4 c = texture(u_src, v_uv) * 0.2270270270;
    c += (texture(u_src, v_uv + u_dir * 1.3846153846) + texture(u_src, v_uv - u_dir * 1.3846153846)) * 0.3162162162;
    c += (texture(u_src, v_uv + u_dir * 3.2307692308) + texture(u_src, v_uv - u_dir * 3.2307692308)) * 0.0702702703;
    f = c;
}
'''
COMPOSITE_FS = r'''
#version 330
uniform sampler2D u_scene; uniform sampler2D u_blurA; uniform sampler2D u_bloomB; uniform sampler2D u_bloomC;
uniform sampler2D u_title;
uniform float u_bloom, u_dof, u_fade, u_time, u_title_amt, u_ca;
uniform vec4 u_title_rect;
in vec2 v_uv; out vec4 f;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec3 tonemap(vec3 x){
    vec3 s = vec3(0.75) + 0.25 * (1.0 - exp(-(x - 0.75) / 0.25));
    return mix(x, s, step(vec3(0.75), x));
}
void main(){
    vec2 uv = v_uv;
    vec2 c = uv - 0.5;
    vec2 off = c * dot(c, c) * 0.010 * u_ca;
    vec4 s0 = texture(u_scene, uv);
    vec3 sc = vec3(texture(u_scene, uv + off).r, s0.g, texture(u_scene, uv - off).b);
    vec3 col = mix(sc, texture(u_blurA, uv).rgb, smoothstep(0.08, 0.9, s0.a) * u_dof);
    col += (texture(u_bloomB, uv).rgb * 0.65 + texture(u_bloomC, uv).rgb * 0.95) * u_bloom;
    col = tonemap(max(col, 0.0));
    float vig = 1.0 - 0.55 * smoothstep(0.35, 0.95, length(c * vec2(1.0, 1.15)) * 1.25);
    col *= vig;
    col = pow(col, vec3(1.0 / 2.2));
    col += (hash(uv * 1280.0 + fract(u_time) * 97.0) - 0.5) * 0.020;
    // title
    float shade = smoothstep(0.30, 0.0, uv.y) * 0.55 * u_title_amt;
    col *= 1.0 - shade;
    vec2 tuv = (uv - u_title_rect.xy) / (u_title_rect.zw - u_title_rect.xy);
    if (tuv.x >= 0.0 && tuv.x <= 1.0 && tuv.y >= 0.0 && tuv.y <= 1.0) {
        vec4 T = texture(u_title, tuv);
        col = mix(col, T.rgb, T.a * u_title_amt);
    }
    col *= u_fade;
    f = vec4(clamp(col, 0.0, 1.0), 1.0);
}
'''


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
def setu(prog, name, value):
    try:
        u = prog[name]
    except KeyError:
        return
    if isinstance(value, (bytes, bytearray)):
        u.write(value)
    elif isinstance(value, np.ndarray):
        u.write(value.astype("f4").tobytes())
    else:
        u.value = value


def build_box():
    faces = [
        ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], 0.0),
        ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)], 2.0),
        ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)], 1.0),
        ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)], 1.0),
        ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)], 1.0),
        ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)], 1.0),
    ]
    out = []
    for n, quad, kind in faces:
        for idx in (0, 1, 2, 0, 2, 3):
            x, y, z = quad[idx]
            out.append([x, y, z, n[0], n[1], n[2], (x + 1) / 2 if kind == 0 else 0, (1 - y) / 2 if kind == 0 else 0, kind])
    return np.array(out, np.float32)


class Renderer:
    def __init__(self, ctx, assets, scene, size, quality, out_fbo=None, scale=None, post=True):
        import moderngl
        self.mgl = moderngl
        self.ctx = ctx
        self.assets = assets
        self.scene = scene
        self.quality = quality
        self.q = QUALITY[quality]
        self.scale = self.q["scale"] if scale is None else scale
        self.post = post
        self.out_fbo = out_fbo
        self.size = size
        self.rig = CameraRig(scene)
        self.title_img = make_title_texture()
        self._build_programs()
        self._build_static()
        self.set_scene(scene)
        self._build_targets()
        self.first = True

    # ---------------------------------------------------------------
    def _build_programs(self):
        c = self.ctx
        self.p_screen = c.program(vertex_shader=SCREEN_VS, fragment_shader=SCREEN_FS)
        self.p_dust = c.program(vertex_shader=DUST_VS, fragment_shader=DUST_FS)
        self.p_halo = c.program(vertex_shader=HALO_VS, fragment_shader=HALO_FS)
        self.p_accum = c.program(vertex_shader=FS_VS, fragment_shader=ACCUM_FS)
        self.p_down = c.program(vertex_shader=FS_VS, fragment_shader=DOWN_FS)
        self.p_blur = c.program(vertex_shader=FS_VS, fragment_shader=BLUR_FS)
        self.p_comp = c.program(vertex_shader=FS_VS, fragment_shader=COMPOSITE_FS)
        self.vao_accum = c.vertex_array(self.p_accum, [])
        self.vao_down = c.vertex_array(self.p_down, [])
        self.vao_blur = c.vertex_array(self.p_blur, [])
        self.vao_comp = c.vertex_array(self.p_comp, [])

    def _build_static(self):
        c, mgl = self.ctx, self.mgl
        a = self.assets
        # photo atlas (mip-mapped, anisotropic)
        img = np.ascontiguousarray(a["atlas_img"])
        self.atlas = c.texture((img.shape[1], img.shape[0]), 3, img.tobytes())
        self.atlas.build_mipmaps(0, 3)
        self.atlas.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        self.atlas.anisotropy = 8.0
        self.atlas.repeat_x = self.atlas.repeat_y = False
        # reference portrait (sampled only for the final colour lock)
        tg = np.ascontiguousarray(a["target"])
        self.target_tex = c.texture((tg.shape[1], tg.shape[0]), 3, tg.tobytes())
        self.target_tex.build_mipmaps()
        self.target_tex.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        self.target_tex.repeat_x = self.target_tex.repeat_y = False
        self.target_size = (float(tg.shape[1]), float(tg.shape[0]))
        # title
        ti = np.ascontiguousarray(self.title_img[::-1])
        self.title_tex = c.texture((ti.shape[1], ti.shape[0]), 4, ti.tobytes())
        self.title_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self.title_aspect = ti.shape[1] / ti.shape[0]
        # geometry
        self.box_vbo = c.buffer(build_box().tobytes())
        self.halo_vbo = c.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        self.vao_halo = c.vertex_array(self.p_halo, [(self.halo_vbo, "2f", "in_p")])
        self._build_dust()

    def _build_dust(self):
        rng = np.random.default_rng(5)
        n = self.q["dust"]
        base = np.stack([rng.uniform(-40, 40, n), rng.uniform(-22, 22, n), rng.uniform(-40, 40, n)], axis=1)
        seed = rng.random((n, 4))
        data = np.concatenate([base, seed], axis=1).astype(np.float32)
        if hasattr(self, "dust_vbo"):
            self.dust_vbo.release()
        self.dust_vbo = self.ctx.buffer(data.tobytes())
        self.dust_n = n
        self.vao_dust = self.ctx.vertex_array(self.p_dust, [(self.dust_vbo, "3f 4f", "in_base", "in_seed")])

    def set_scene(self, scene):
        c = self.ctx
        self.scene = scene
        self.rig = CameraRig(scene)
        for name in ("dyn_vbo", "stat_vbo"):
            if hasattr(self, name):
                getattr(self, name).release()
        stat = np.concatenate([scene.tile_rect, scene.tile_mean, scene.tgt_rect], axis=1).astype(np.float32)
        self.stat_vbo = c.buffer(stat.tobytes())
        self.dyn_vbo = c.buffer(reserve=scene.n * 14 * 4, dynamic=True)
        self.vao_screen = c.vertex_array(self.p_screen, [
            (self.box_vbo, "3f 3f 2f 1f", "in_pos", "in_normal", "in_uv", "in_kind"),
            (self.dyn_vbo, "3f 4f 3f 4f/i", "i_pos", "i_quat", "i_half", "i_state"),
            (self.stat_vbo, "4f 3f 4f/i", "i_tile", "i_mean", "i_tgt"),
        ])
        self.first = True

    # ---------------------------------------------------------------
    def _tex(self, w, h, dtype="f2", samples=0):
        t = self.ctx.texture((max(1, w), max(1, h)), 4, dtype=dtype, samples=samples)
        if not samples:
            t.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
            t.repeat_x = t.repeat_y = False
        return t

    def _build_targets(self):
        c = self.ctx
        W, H = self.size
        self.iw = max(64, int(W * self.scale))
        self.ih = max(64, int(H * self.scale))
        iw, ih = self.iw, self.ih
        self.msaa = self.q["msaa"]
        self.scene_fbo = None
        if self.msaa:
            try:
                self.scene_tex = self._tex(iw, ih, samples=self.msaa)
                depth = c.depth_renderbuffer((iw, ih), samples=self.msaa)
                self.scene_fbo = c.framebuffer([self.scene_tex], depth)
                self.resolve_tex = self._tex(iw, ih)
                self.resolve_fbo = c.framebuffer([self.resolve_tex])
            except Exception as exc:
                print("[mosaic] MSAA unavailable, falling back:", str(exc)[:80])
                self.msaa = 0
        if not self.msaa:
            self.scene_tex = self._tex(iw, ih)
            depth = c.depth_renderbuffer((iw, ih))
            self.scene_fbo = c.framebuffer([self.scene_tex], depth)
            self.resolve_tex, self.resolve_fbo = self.scene_tex, self.scene_fbo
        self.acc = [self._tex(iw, ih), self._tex(iw, ih)]
        self.acc_fbo = [c.framebuffer([t]) for t in self.acc]
        self.acc_i = 0
        sizes = [(iw // 2, ih // 2), (iw // 4, ih // 4), (iw // 8, ih // 8)]
        self.chain = []
        for (w, h) in sizes:
            ta, tb = self._tex(w, h), self._tex(w, h)
            self.chain.append(dict(size=(max(1, w), max(1, h)), a=ta, b=tb,
                                   fa=c.framebuffer([ta]), fb=c.framebuffer([tb])))
        self.first = True

    def resize(self, size, scale=None):
        self.size = size
        if scale is not None:
            self.scale = scale
        self._build_targets()

    def set_quality(self, quality):
        self.quality = quality
        self.q = QUALITY[quality]
        self.scale = self.q["scale"]
        self._build_dust()
        self._build_targets()

    # ---------------------------------------------------------------
    def _fs_pass(self, vao, prog, fbo, textures, uniforms):
        mgl = self.mgl
        fbo.use()
        self.ctx.disable(mgl.DEPTH_TEST | mgl.BLEND)
        for i, (name, tex) in enumerate(textures.items()):
            tex.use(location=i)
            setu(prog, name, i)
        for k, v in uniforms.items():
            setu(prog, k, v)
        vao.render(mgl.TRIANGLES, vertices=3)

    def render(self, t, dt, reset_trails=False):
        mgl, ctx = self.mgl, self.ctx
        W, H = self.size
        aspect = W / H
        eye, look, up, dist = self.rig.pose(t, aspect)
        view = look_at(eye, look, up)
        proj = perspective(FOV, aspect, 0.5, 220.0)
        vp = proj @ view
        vp_b = vp.T.astype("f4").tobytes()
        view_b = view.T.astype("f4").tobytes()

        self.dyn_vbo.write(self.scene.update(t).tobytes())

        # --- timeline-driven look parameters ---
        aperture = 0.75 - 0.30 * float(smooth((t - 3.0) / 3.0)) - 0.20 * float(smooth((t - 8.0) / 4.0)) - 0.10 * float(smooth((t - 12.0) / 2.0))
        fog = 0.013 * (1 - float(smooth((t - 2.0) / 6.0))) + 0.0025
        gloss = 1.0 - 0.45 * float(smooth((t - 11.0) / 4.0))
        edge = 1.0 + 0.8 * (1 - float(smooth((t - 3.0) / 3.0)))
        sweep = -26.0 + 52.0 * float(smooth((t - 14.0) / 2.8))
        sweep_amt = float(smooth((t - 13.8) / 0.5) * (1 - smooth((t - 16.6) / 0.6)))
        halo = float(smooth((t - 12.3) / 4.2))
        title_amt = float(smooth((t - T_REVEAL_END) / 1.1))
        bloom = 0.30 + 0.12 * float(smooth((t - 3.0) / 3.0)) + 0.12 * halo
        fade = float(smooth(t / 1.4))
        decay = 0.62 * float(smooth((t - 2.5) / 2.0)) * (1 - float(smooth((t - 11.5) / 2.0)))
        decay = decay ** (max(dt, 1e-3) * 60.0) if decay > 0 else 0.0
        if reset_trails or self.first:
            decay = 0.0
        lpos = np.array([[-18 + 6 * math.sin(.3 * t), 10 + 3 * math.cos(.2 * t), 14],
                         [18 + 5 * math.cos(.25 * t), -6 + 4 * math.sin(.31 * t), 12],
                         [4 * math.sin(.17 * t), 16, 10 + 4 * math.cos(.13 * t)]], np.float32)
        lcol = np.array([[0.35, 0.60, 1.30], [1.30, 0.85, 0.35], [1.0, 1.0, 1.0]], np.float32)

        # --- 3D pass -----------------------------------------------------
        fbo = self.scene_fbo
        fbo.use()
        ctx.viewport = (0, 0, self.iw, self.ih)
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND | mgl.CULL_FACE)
        pr = self.p_screen
        self.atlas.use(location=0)
        self.target_tex.use(location=1)
        setu(pr, "u_atlas", 0)
        setu(pr, "u_target", 1)
        for k, v in dict(u_viewproj=vp_b, u_view=view_b, u_cam=tuple(float(x) for x in eye),
                         u_target_size=self.target_size, u_time=t, u_fog=fog, u_aperture=aperture,
                         u_focus=float(dist), u_k=DETAIL_K, u_gloss=gloss, u_sweep=sweep,
                         u_sweep_amt=sweep_amt, u_edge=edge, u_lpos=lpos, u_lcol=lcol).items():
            setu(pr, k, v)
        self.vao_screen.render(mgl.TRIANGLES, vertices=36, instances=self.scene.n)

        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE, mgl.ZERO, mgl.ONE)
        fbo.depth_mask = False
        if halo > 0.001:
            ph = self.p_halo
            setu(ph, "u_viewproj", vp_b)
            setu(ph, "u_ext", (self.scene.Wp / 2 + 90.0, self.scene.Hp / 2 + 90.0))
            setu(ph, "u_half", (self.scene.Wp / 2, self.scene.Hp / 2))
            setu(ph, "u_amt", 0.75 * halo)
            setu(ph, "u_time", t)
            self.vao_halo.render(mgl.TRIANGLE_STRIP)
        dust_amt = 0.55 * float(smooth(t / 2.0)) + 0.45 * halo
        if dust_amt > 0.001:
            ctx.enable(mgl.PROGRAM_POINT_SIZE)
            pd = self.p_dust
            tan_half = math.tan(math.radians(FOV / 2))
            for k, v in dict(u_viewproj=vp_b, u_view=view_b, u_time=t,
                             u_pxscale=self.ih / tan_half * 0.5, u_focus=float(dist), u_alpha=dust_amt).items():
                setu(pd, k, v)
            self.vao_dust.render(mgl.POINTS)
        fbo.depth_mask = True
        ctx.disable(mgl.BLEND)
        ctx.disable(mgl.DEPTH_TEST)

        if self.msaa:
            ctx.copy_framebuffer(self.resolve_fbo, self.scene_fbo)

        # --- temporal trails ---------------------------------------------
        prev = self.acc[self.acc_i]
        self.acc_i ^= 1
        cur_fbo = self.acc_fbo[self.acc_i]
        self._fs_pass(self.vao_accum, self.p_accum, cur_fbo,
                      {"u_cur": self.resolve_tex, "u_prev": prev}, {"u_decay": decay})
        scene_tex = self.acc[self.acc_i]

        # --- blur / bloom chain ------------------------------------------
        if self.post:
            A, B, Cc = self.chain
            self._fs_pass(self.vao_down, self.p_down, A["fa"], {"u_src": scene_tex},
                          {"u_texel": (1.0 / self.iw, 1.0 / self.ih), "u_thr": 0.0, "u_use_thr": 0.0})
            self._blur(A)
            self._fs_pass(self.vao_down, self.p_down, B["fa"], {"u_src": A["a"]},
                          {"u_texel": (1.0 / A["size"][0], 1.0 / A["size"][1]), "u_thr": 0.62, "u_use_thr": 1.0})
            self._blur(B)
            self._fs_pass(self.vao_down, self.p_down, Cc["fa"], {"u_src": B["a"]},
                          {"u_texel": (1.0 / B["size"][0], 1.0 / B["size"][1]), "u_thr": 0.0, "u_use_thr": 0.0})
            self._blur(Cc)

        # --- composite to the screen -------------------------------------
        out = self.out_fbo if self.out_fbo is not None else ctx.detect_framebuffer()
        out.use()
        ctx.viewport = (0, 0, W, H)
        th_frac = 0.066
        tw_frac = th_frac * self.title_aspect * H / W
        rect = (0.5 - tw_frac / 2, 0.070 - th_frac / 2, 0.5 + tw_frac / 2, 0.070 + th_frac / 2)
        A, B, Cc = self.chain
        self._fs_pass(self.vao_comp, self.p_comp, out,
                      {"u_scene": scene_tex, "u_blurA": A["a"], "u_bloomB": B["a"],
                       "u_bloomC": Cc["a"], "u_title": self.title_tex},
                      {"u_bloom": bloom if self.post else 0.0, "u_dof": 0.9 if self.post else 0.0,
                       "u_fade": fade, "u_time": t, "u_title_amt": title_amt,
                       "u_ca": 1.0 if self.post else 0.0, "u_title_rect": rect})
        self.first = False

    def _blur(self, lvl):
        w, h = lvl["size"]
        self._fs_pass(self.vao_blur, self.p_blur, lvl["fb"], {"u_src": lvl["a"]}, {"u_dir": (1.0 / w, 0.0)})
        self._fs_pass(self.vao_blur, self.p_blur, lvl["fa"], {"u_src": lvl["b"]}, {"u_dir": (0.0, 1.0 / h)})


# ----------------------------------------------------------------------------
# Asset preparation (shared by live mode and the offscreen test tool)
# ----------------------------------------------------------------------------
def prepare_assets():
    photos = load_photos()
    target, focus = load_target(photos)
    # tile aspect must equal the real cell aspect of the layout
    A = target.shape[1] / target.shape[0]
    R = int(4 * round(GRID_COLS * CELL_ASPECT / A / 4))
    cell_aspect = A * R / GRID_COLS
    atlas_img, rects, means = build_atlas(photos, cell_aspect)
    print(f"[mosaic] photo atlas: {len(means)} unique tiles cropped from {len(photos)} photo(s)")
    return dict(photos=photos, target=target, focus=focus,
                atlas_img=atlas_img, atlas_rects=rects, atlas_means=means)


# ----------------------------------------------------------------------------
# Live application
# ----------------------------------------------------------------------------
def run_live(args):
    import moderngl
    import pygame

    assets = prepare_assets()
    quality = args.quality
    count = args.count or QUALITY[quality]["count"]
    t0 = time.time()
    scene = Scene(assets, count)
    print(f"[mosaic] {scene.n} screens laid out in {time.time() - t0:.2f}s (quality={quality})")

    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("PAWAN KALYAN - The 1000-Screen Mosaic")
    gl = pygame.display
    gl.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    gl.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    gl.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    gl.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    gl.gl_set_attribute(pygame.GL_DEPTH_SIZE, 0)

    state = dict(fullscreen=args.fullscreen, windowed=(args.width, args.height))

    def make_window():
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if state["fullscreen"]:
            flags |= pygame.FULLSCREEN
            size = (0, 0)
        else:
            flags |= pygame.RESIZABLE
            size = state["windowed"]
        try:
            pygame.display.set_mode(size, flags, vsync=0 if args.no_vsync else 1)
        except TypeError:
            pygame.display.set_mode(size, flags)
        return pygame.display.get_window_size()

    size = make_window()
    ctx = moderngl.create_context()
    print("[mosaic] GL:", ctx.info.get("GL_RENDERER", "?"))
    pygame.mouse.set_visible(not state["fullscreen"])
    renderer = Renderer(ctx, assets, scene, size, quality, scale=args.scale)

    clock = pygame.time.Clock()
    t, paused, post = 0.0, False, True
    fps_acc, fps_n, last_cap = 0.0, 0, time.time()
    slow_time, adapt_clock = 0.0, 0.0
    running = True
    wall0 = time.time()
    script = [(1.0, pygame.K_SPACE), (1.6, pygame.K_SPACE), (2.2, pygame.K_b), (2.8, pygame.K_b),
              (3.4, pygame.K_q), (4.6, pygame.K_f), (6.5, pygame.K_f), (8.0, pygame.K_r)] if args.selftest else []
    while running:
        while script and time.time() - wall0 > script[0][0]:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=script.pop(0)[1]))
        if args.max_seconds and time.time() - wall0 > args.max_seconds:
            break
        dt = min(clock.tick(0 if args.no_vsync else 120) / 1000.0, 0.1)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    t = 0.0
                    renderer.first = True
                elif ev.key == pygame.K_b:
                    post = not post
                    renderer.post = post
                elif ev.key == pygame.K_q:
                    quality = QUALITY_ORDER[(QUALITY_ORDER.index(quality) + 1) % 3]
                    scene = Scene(assets, QUALITY[quality]["count"])
                    renderer.set_quality(quality)
                    renderer.set_scene(scene)
                    print(f"[mosaic] quality -> {quality} ({scene.n} screens)")
                elif ev.key == pygame.K_f:
                    if not state["fullscreen"]:
                        state["windowed"] = pygame.display.get_window_size()
                    state["fullscreen"] = not state["fullscreen"]
                    size = make_window()
                    ctx = moderngl.create_context()
                    pygame.mouse.set_visible(not state["fullscreen"])
                    renderer = Renderer(ctx, assets, scene, size, quality, scale=renderer.scale, post=post)
        cur = pygame.display.get_window_size()
        if cur != renderer.size and cur[0] > 0 and cur[1] > 0:
            renderer.resize(cur)
        if not paused:
            t += dt
        renderer.render(t, dt)
        pygame.display.flip()

        # lightweight adaptive resolution: keep the animation smooth on weak GPUs
        if args.adaptive and not paused and t > 0.5:
            adapt_clock += dt
            slow_time += dt if dt > 1 / 24 else -dt * 0.5
            slow_time = max(slow_time, 0.0)
            if slow_time > 1.2 and renderer.scale > 0.55:
                renderer.resize(renderer.size, round(renderer.scale - 0.1, 2))
                print(f"[mosaic] low frame-rate: render scale -> {renderer.scale:.2f}")
                slow_time = 0.0
        fps_acc += dt
        fps_n += 1
        if time.time() - last_cap > 0.5:
            fps = fps_n / max(fps_acc, 1e-6)
            pygame.display.set_caption(
                f"PAWAN KALYAN - The 1000-Screen Mosaic | {fps:5.1f} fps | {scene.n} screens | "
                f"{quality} | t={t:4.1f}s{' (paused)' if paused else ''}")
            fps_acc, fps_n, last_cap = 0.0, 0, time.time()
    pygame.quit()


# ----------------------------------------------------------------------------
# Offscreen test tool (not used in normal operation): renders chosen moments
# to PNG so the result can be inspected on machines without a display.
# ----------------------------------------------------------------------------
def run_dump(args):
    import moderngl
    assets = prepare_assets()
    quality = args.quality
    count = args.count or QUALITY[quality]["count"]
    scene = Scene(assets, count)
    print(f"[dump] {scene.n} screens")
    try:
        ctx = moderngl.create_standalone_context(backend="egl")
    except Exception:
        ctx = moderngl.create_standalone_context()
    w, h = args.width, args.height
    tex = ctx.texture((w, h), 4)
    fbo = ctx.framebuffer([tex])
    renderer = Renderer(ctx, assets, scene, (w, h), quality, out_fbo=fbo, scale=args.scale)
    outdir = Path(args.dump)
    outdir.mkdir(parents=True, exist_ok=True)
    times = [float(x) for x in args.times.split(",")]
    for tt in times:
        # a few lead-in frames so temporal trails look like they do live
        for k in range(args.warm, 0, -1):
            renderer.render(max(tt - k / 30.0, 0.0), 1 / 30.0)
        t0 = time.time()
        renderer.render(tt, 1 / 30.0, reset_trails=args.warm == 0)
        ctx.finish()
        data = np.frombuffer(fbo.read(components=3), np.uint8).reshape(h, w, 3)[::-1]
        path = outdir / f"frame_{tt:05.2f}.png"
        cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(data), cv2.COLOR_RGB2BGR))
        print(f"[dump] t={tt:5.2f}s -> {path.name}  ({time.time() - t0:.2f}s)")


def main():
    ap = argparse.ArgumentParser(description="The 1000-Screen Mosaic - live cinematic animation")
    ap.add_argument("--quality", choices=QUALITY_ORDER, default="medium", help="low / medium / high")
    ap.add_argument("--count", type=int, default=0, help="override the number of screens")
    ap.add_argument("--scale", type=float, default=None, help="internal render scale (0.5-1.0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-vsync", action="store_true")
    ap.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="disable automatic resolution scaling")
    ap.add_argument("--max-seconds", type=float, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dump", default="", help="(testing) render frames offscreen into this folder")
    ap.add_argument("--times", default="1,5,9,12,15,18", help="(testing) comma separated seconds to dump")
    ap.add_argument("--warm", type=int, default=0, help="(testing) lead-in frames before each dump")
    args = ap.parse_args()
    if args.dump:
        run_dump(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
