import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_closing, label

# ──────────────────────────────────────────────
# 1. GLOBAL CONSTANTS
# ──────────────────────────────────────────────
GRID        = 256   # simulation grid (px)
VIEW_PIXELS = 250   # template window on GUI (px)
WAVELENGTHS = {"EUV": 13.5e-3, "ArF": 193e-3}  # µm
NAS         = {"EUV": 0.33,    "ArF": 0.85}
SQRT2       = np.sqrt(2.0)

# ──────────────────────────────────────────────
# 2. Utilities
# ──────────────────────────────────────────────
px = lambda nm, nm_per_px: max(1, int(round(nm / nm_per_px)))

def diag_centers(n, pitch, cx, cy):
    step, off = pitch / SQRT2, -(n // 2)
    for i in range(n):
        for j in range(n):
            u, v = off + i, off + j
            yield int(cx + round((u - v) * step)), int(cy + round((u + v) * step))

# ──────────────────────────────────────────────
# 3. DESIGN (no OPC)
# ──────────────────────────────────────────────

def make_design(sel, size_nm, pitch_nm, tmpl_nm, nm_per_px):
    """Return base DESIGN pattern only."""
    typ  = "Via" if "Via" in sel else "Line"
    diag = sel == "Diagonal Via"

    design = np.zeros((GRID, GRID), float)
    cx = cy = GRID // 2
    size_px  = px(size_nm, nm_per_px)
    pitch_px = px(pitch_nm, nm_per_px)
    tmpl_px  = px(tmpl_nm, nm_per_px)

    if typ == "Via":
        half = size_px // 2
        n = max(1, (tmpl_px // pitch_px) | 1)
        centers = diag_centers(n, pitch_px, cx, cy) if diag else (
            (cx + (i - n//2) * pitch_px, cy + (j - n//2) * pitch_px)
            for i in range(n) for j in range(n))
        for x0, y0 in centers:
            design[y0-half:y0+half, x0-half:x0+half] = 1
    else:
        line_w = size_px
        half_t = tmpl_px // 2
        y_margin = int(0.10 * tmpl_px)
        y0 = cy - half_t + y_margin
        y1 = cy + half_t - y_margin
        for offset in range(0, half_t + pitch_px, pitch_px):
            sx_list = [0] if offset == 0 else [-offset, offset]
            for sx in sx_list:
                x0 = cx + sx - line_w // 2
                x1 = x0 + line_w
                if 0 <= x0 < GRID and x1 < GRID:
                    design[y0:y1, x0:x1] = 1
    return design, typ

# ──────────────────────────────────────────────
# 4. MASK & OPC
# ──────────────────────────────────────────────

def to_mask(design, tone, resist):
    mask = 1 - design if tone == "Bright" else design.copy()
    return 1 - mask if resist == "NTD" else mask


def opc_rule(mask, typ, add_serif=True):
    """Rule OPC: 1‑px bias + serifs / hammer‑head."""
    opc = binary_dilation(mask.astype(bool), iterations=1).astype(float)
    THK = 10  # hammer‑head & serif thickness (px)

    if typ == "Line":
        # hammer‑head: extend 10‑px at top / bottom
        opc[:THK] = 1; opc[-THK:] = 1
        comp, n = label(mask.astype(bool))
        for idx in range(1, n+1):
            ys, xs = np.where(comp == idx)
            if xs.size == 0: continue
            minx, maxx = xs.min(), xs.max(); miny, maxy = ys.min(), ys.max()
            # 10‑px corner serifs at each bar end
            for y, x in [
                (miny - THK, minx - THK),
                (maxy + 1,   minx - THK),
                (miny - THK, maxx + 1),
                (maxy + 1,   maxx + 1),
            ]:
                opc[max(0, y):min(GRID, y + THK), max(0, x):min(GRID, x + THK)] = 1
    elif typ == "Via" and add_serif:
        comp, n = label(mask.astype(bool))
        for idx in range(1, n+1):
            ys, xs = np.where(comp == idx)
            if xs.size == 0: continue
            minx, maxx = xs.min(), xs.max(); miny, maxy = ys.min(), ys.max()
            for y, x in [
                (miny - int(THK/2), minx - int(THK/2)),
                (maxy + 1,   minx - int(THK/2)),
                (miny - int(THK/2), maxx + 1),
                (maxy + 1,   maxx + 1),
            ]:
                opc[max(0, y):min(GRID, y + int(THK/2)), max(0, x):min(GRID, x + int(THK/2))] = 1
    return opc


def opc_curvi(mask, typ, size_px):
    if typ == "Via":
        radius = int(size_px / 1.3)
        comp, n = label(mask.astype(bool))
        Y, X = np.ogrid[:GRID, :GRID]
        out = np.zeros_like(mask)
        for idx in range(1, n+1):
            ys, xs = np.where(comp == idx)
            if xs.size == 0: continue
            cx, cy = int(xs.mean()), int(ys.mean())
            out[(X - cx)**2 + (Y - cy)**2 <= radius**2] = 1
        return out.astype(float)
    else:
        line = binary_closing(mask.astype(bool), structure=np.ones((3, 7), bool))
        st = np.array([[0,1,0],[1,1,1],[0,1,0]], bool)
        return binary_closing(mask.astype(bool), structure=st).astype(float)


def opc_sraf(design, mask, typ, pitch_px, size_px):
    """Add scattering bars.
    Line  → thin bars at ±0.3·pitch, limited to two horizontal bands (≈20 % height) so they print faint.
    Via   → small assist circles (unchanged).
    """
    opc = mask.copy()
    if typ == "Line":
        off    = int(0.30 * pitch_px)
        bar_w  = max(1, size_px // 4)   # much thinner
        # narrow y bands (±25 px around center)
        band = 80
        rows = slice(GRID//2-band, GRID//2+band)
        # locate mid‑x of each main bar (contiguous group)
        cols = np.where(mask[GRID//2] == 1)[0]
        if cols.size:
            groups = np.split(cols, np.where(np.diff(cols) != 1)[0] + 1)
            centers = [g[len(g)//2] for g in groups]
            for cx in centers:
                for dx in (-off, off):
                    x0 = cx + dx - bar_w//2
                    x1 = x0 + bar_w
                    if 0 <= x0 < GRID and x1 < GRID:
                        opc[rows, x0:x1] = 1
    else:
        radius_via = int(size_px / 1.3)
        comp, n = label(mask.astype(bool))
        Y, X = np.ogrid[:GRID, :GRID]
        opc = np.zeros_like(mask)
        for idx in range(1, n+1):
            ys, xs = np.where(comp == idx)
            if xs.size == 0: continue
            cx, cy = int(xs.mean()), int(ys.mean())
            opc[(X - cx)**2 + (Y - cy)**2 <= radius_via**2] = 1
        radius_sraf = max(1, size_px // 3)
        off = int(0.25 * (pitch_px - size_px))
        ys, xs = np.where(mask)
        if xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            for dx, dy in [(-off,-off),(off,-off),(-off,off),(off,off)]:
                opc[(X - cx - dx)**2 + (Y - cy - dy)**2 <= radius_sraf**2] = 1
    return opc

# ──────────────────────────────────────────────
# 5. Optics simulator 
# ──────────────────────────────────────────────

def make_pupil(FX, FY, na, wl, illum):
    k = na / wl; R = np.hypot(FX, FY)
    if illum == "Conventional":
        return (R <= k).astype(float)
    if illum == "Annular":
        return ((R <= k) & (R >= 0.6 * k)).astype(float)
    if illum == "Quadrupole":
        P = np.zeros_like(R, bool); pr, pc = 0.15 * k, 0.7 * k
        for cx, cy in [(pc,0),(-pc,0),(0,pc),(0,-pc)]:
            P |= ((FX-cx)**2 + (FY-cy)**2) <= pr**2
        return P.astype(float)
    return (R <= k).astype(float)


def make_simulator(wl, na, illum, umpp):
    f = np.fft.fftshift(np.fft.fftfreq(GRID, d=umpp))
    FX, FY = np.meshgrid(f, f)
    P = make_pupil(FX, FY, na, wl, illum)
    def sim(mask):
        FM = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(FM * P)))
        return np.abs(img)**2  # raw intensity (no auto‑norm)
    return sim

# ──────────────────────────────────────────────
# 6. Save helpers
# ──────────────────────────────────────────────

def save_img(a, path, cmap="gray", ttl=""):
    plt.figure(figsize=(3,3)); plt.imshow(a, cmap=cmap, origin="lower")
    plt.title(ttl); plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()

def save_profile(I, path):
    mid = I.shape[0]//2
    plt.figure(figsize=(3,3)); plt.plot(I[mid]); plt.ylim(0,1)
    plt.title("Center X profile"); plt.grid(); plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()

# ──────────────────────────────────────────────
# 7. GUI
# ──────────────────────────────────────────────
class OPCDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPC Demo – SRAF")
        self.resize(1120, 440)
        v = QVBoxLayout(self)

        # ── Controls
        self.src   = QComboBox(); self.src.addItems(["EUV", "ArF"])
        self.pat   = QComboBox(); self.pat.addItems(["Orthogonal Via", "Diagonal Via", "Line"])
        self.tone  = QComboBox(); self.tone.addItems(["Dark", "Bright"])
        self.res   = QComboBox(); self.res.addItems(["PTD", "NTD"])
        self.illum = QComboBox(); self.illum.addItems(["Conventional", "Annular", "Quadrupole"])
        self.opc   = QComboBox(); self.opc.addItems(["No OPC", "Rule", "Curvi", "SRAF"])
        self.size  = QLineEdit("30")
        self.pitch = QLineEdit("150")
        self.tmpl  = QLineEdit("500")

        def add_row(label, widget):
            h = QHBoxLayout(); h.addWidget(QLabel(label)); h.addWidget(widget); v.addLayout(h)

        for lbl, w in [
            ("Source",       self.src),
            ("Pattern",      self.pat),
            ("Mask Tone",    self.tone),
            ("Resist",       self.res),
            ("Illumination", self.illum),
            ("OPC Method",   self.opc),
            ("Feature [nm]", self.size),
            ("Pitch [nm]",   self.pitch),
            ("Template [nm]",self.tmpl),
        ]:
            add_row(lbl, w)

        run_btn = QPushButton("Run Simulation"); run_btn.setStyleSheet("font-weight:bold;")
        run_btn.clicked.connect(self.run)
        v.addWidget(run_btn)

        # ── Image placeholders
        self.views = [QLabel() for _ in range(4)]
        h_img = QHBoxLayout()
        for lab in self.views:
            lab.setFixedSize(260, 260)
            h_img.addWidget(lab)
        v.addLayout(h_img)

    # ────────────────────────────────────────
    def run(self):
        # 1. parse inputs
        wl = WAVELENGTHS[self.src.currentText()]
        na = NAS[self.src.currentText()]
        try:
            size_nm  = float(self.size.text())
            pitch_nm = float(self.pitch.text())
            tmpl_nm  = float(self.tmpl.text())
        except ValueError:
            return

        nmpp = tmpl_nm / VIEW_PIXELS; umpp = nmpp * 1e-3

        # 2. design & mask
        mask, typ = make_design(self.pat.currentText(), size_nm, pitch_nm, tmpl_nm, nmpp)
        sim    = make_simulator(wl, na, self.illum.currentText(), umpp)

        # 3. OPC selection
        choice = self.opc.currentText()
        if choice == "No OPC":
            mask_opc = mask.copy()
        elif choice == "Rule":
            mask_opc = opc_rule(mask, typ, add_serif=True)
        elif choice == "Curvi":
            mask_opc = opc_curvi(mask, typ, px(size_nm, nmpp))
        else:  # SRAF
            pitch_px = px(pitch_nm, nmpp)
            size_px  = px(size_nm,  nmpp)
            mask_opc = opc_sraf(mask, mask, typ, pitch_px, size_px)  # ← tone 인자 제거

        mask_tone   = to_mask(mask_opc, self.tone.currentText(), self.res.currentText())
        
        # 4. Aerial image (normalize to No‑OPC max)
        aerial = sim(mask_tone)

        # 5. Save + display
        save_img(mask,  "design.png", ttl="Design")
        save_img(mask_tone, "opc.png",    ttl=f"OPC ({choice})")
        save_img(aerial,   "aerial.png", cmap="inferno", ttl="Aerial")
        save_profile(aerial, "profile.png")

        for path, lab in zip(["design.png", "opc.png", "aerial.png", "profile.png"], self.views):
            lab.setPixmap(QPixmap(path).scaled(250,250,Qt.KeepAspectRatio,Qt.SmoothTransformation))

  # ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OPCDemo(); win.show()
    sys.exit(app.exec_())
