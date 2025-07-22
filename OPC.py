import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter

# ──────────────────────────────────────────────
# 1. GLOBAL CONSTANTS
# ──────────────────────────────────────────────
GRID           = 256            # simulation resolution (px)
VIEW_PIXELS    = 250            # template window on GUI (px)
WAVELENGTHS    = {"EUV": 13.5e-3, "ArF": 193e-3}  # microns
NAS            = {"EUV": 0.33,    "ArF": 0.85}
SQRT2          = np.sqrt(2.0)

# ──────────────────────────────────────────────
# 2. utilities
# ──────────────────────────────────────────────
px = lambda nm, nm_per_px: max(1, int(round(nm / nm_per_px)))

def diag_centers(n, pitch, cx, cy):
    step, off = pitch / SQRT2, -(n // 2)
    for i in range(n):
        for j in range(n):
            u, v = off + i, off + j
            yield int(round(cx + (u - v) * step)), int(round(cy + (u + v) * step))

# ──────────────────────────────────────────────
# 3. Design generation
# ──────────────────────────────────────────────

def make_design(sel, size_nm, pitch_nm, tmpl_nm, nm_per_px):
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
            for i in range(n) for j in range(n)
        )
        for x, y in centers:
            if half <= x < GRID - half and half <= y < GRID - half:
                design[y-half:y+half, x-half:x+half] = 1
    else:  # Line pattern
        line_w = size_px
        start, end = cx - tmpl_px // 2, cx + tmpl_px // 2
        x = start + line_w // 2
        while x < end:
            design[:, x - line_w // 2 : x + (line_w + 1)//2] = 1
            x += pitch_px
        design[:GRID // 5] = 0  # cut top 20 % to highlight hammer-head
    return design, typ

# ──────────────────────────────────────────────
# 4. Mask & OPC
# ──────────────────────────────────────────────

def to_mask(design, tone, resist):
    mask = 1 - design if tone == "Bright" else design.copy()
    return 1 - mask if resist == "NTD" else mask


def opc_rule(mask, typ):
    opc = binary_dilation(mask, iterations=1)
    if typ == "Via":
        opc |= binary_dilation(mask, iterations=2)  # 2-px serif
    else:
        opc[:4] = 1; opc[-4:] = 1                  # hammer-head
    return opc


def opc_curvi(mask):
    # 부드러움 ↑ : sigma 3 , threshold 0.2
    smooth = gaussian_filter(mask, sigma=3.0)
    return (smooth > 0.20).astype(float)


def opc_ilt(design, init, sim, iters=15):
    mask = init.copy()

    # 디자인 경계 ±6 px 안쪽 ROI 만 후보로 제한
    roi = binary_dilation(design, iterations=6)

    for _ in range(iters):
        diff = (sim(mask) > 0.5).astype(float) - design
        idx  = np.argwhere((diff != 0) & roi)

        if not len(idx):
            break

        # error 크기순 우선 flip → 점박이 감소
        err_vals = np.abs(diff[idx[:, 0], idx[:, 1]])
        idx = idx[err_vals.argsort()[::-1]][:30]   # 상위 30 px
        for y, x in idx:
            mask[y, x] = 1 - mask[y, x]

    return mask


# ──────────────────────────────────────────────
# 5. Optics simulator
# ──────────────────────────────────────────────

def make_pupil(FX, FY, na, wl, illum):
    k = na / wl
    R = np.hypot(FX, FY)

    if illum == "Conventional":
        return (R <= k).astype(float)

    if illum == "Annular":
        return ((R <= k) & (R >= 0.6 * k)).astype(float)

    if illum == "Quadrupole":
        # Boolean mask first, then convert → float
        P = np.zeros_like(R, dtype=bool)
        pr, pc = 0.15 * k, 0.7 * k
        for cx, cy in [(pc, 0), (-pc, 0), (0, pc), (0, -pc)]:
            P |= ((FX - cx) ** 2 + (FY - cy) ** 2) <= pr ** 2
        return P.astype(float)

    return (R <= k).astype(float)



def make_simulator(wl, na, illum, umpp):
    f  = np.fft.fftshift(np.fft.fftfreq(GRID, d=umpp))
    FX, FY = np.meshgrid(f, f)
    P = make_pupil(FX, FY, na, wl, illum)
    def sim(mask):
        FM  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(FM * P)))
        I   = np.abs(img)**2
        return I / I.max()
    return sim

# ──────────────────────────────────────────────
# 6. Save helpers
# ──────────────────────────────────────────────

def save_img(a, path, cmap="gray", ttl=""):
    plt.figure(figsize=(3, 3)); plt.imshow(a, cmap=cmap, origin="lower")
    plt.title(ttl); plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def save_profile(I, path):
    mid = I.shape[0] // 2
    plt.figure(figsize=(3, 3)); plt.plot(I[mid]); plt.ylim(0, 1)
    plt.title("Center X profile"); plt.grid(); plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()

# ──────────────────────────────────────────────
# 7. GUI
# ──────────────────────────────────────────────
class OPCDemo(QWidget):
    """Main GUI window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPC Demo")
        self.resize(1100, 420)

        v = QVBoxLayout(self)

        # ── Control widgets ──
        self.src   = QComboBox(); self.src.addItems(["EUV", "ArF"])
        self.pat   = QComboBox(); self.pat.addItems(["Orthogonal Via", "Diagonal Via", "Line"])
        self.tone  = QComboBox(); self.tone.addItems(["Bright", "Dark"])
        self.res   = QComboBox(); self.res.addItems(["PTD", "NTD"])
        self.illum = QComboBox(); self.illum.addItems(["Conventional", "Annular", "Quadrupole"])
        self.opc   = QComboBox(); self.opc.addItems(["Rule", "Curvi", "ILT"])
        self.size  = QLineEdit("100")
        self.pitch = QLineEdit("200")
        self.tmpl  = QLineEdit("500")

        def add_row(label, widget):
            h = QHBoxLayout(); h.addWidget(QLabel(label)); h.addWidget(widget); v.addLayout(h)

        for lbl, w in [
            ("Source", self.src), ("Pattern", self.pat), ("Mask", self.tone),
            ("Resist", self.res), ("Illum", self.illum), ("OPC", self.opc),
            ("Size nm", self.size), ("Pitch nm", self.pitch), ("Template nm", self.tmpl)
        ]:
            add_row(lbl, w)

        run_btn = QPushButton("Run Simulation")
        run_btn.clicked.connect(self.run)
        v.addWidget(run_btn)

        # ── Image preview labels ──
        self.views = [QLabel() for _ in range(4)]
        img_row = QHBoxLayout()
        for lab in self.views:
            lab.setFixedSize(260, 260)
            img_row.addWidget(lab)
        v.addLayout(img_row)

    # ──────────────────────────────────────────
    # Run button callback
    # ──────────────────────────────────────────
    def run(self):
        # 1. Gather parameters
        wl   = WAVELENGTHS[self.src.currentText()]
        na   = NAS[self.src.currentText()]
        nmpp = float(self.tmpl.text()) / VIEW_PIXELS
        umpp = nmpp * 1e-3

        design, typ = make_design(
            self.pat.currentText(),
            float(self.size.text()),
            float(self.pitch.text()),
            float(self.tmpl.text()),
            nmpp,
        )

        mask = to_mask(design, self.tone.currentText(), self.res.currentText())

        sim = make_simulator(wl, na, self.illum.currentText(), umpp)

        # 2. Apply OPC method
        opc_method = self.opc.currentText()
        if opc_method == "Rule":
            mask_opc = opc_rule(mask, typ)
        elif opc_method == "Curvi":
            mask_opc = opc_curvi(mask)
        else:  # ILT
            mask_opc = opc_ilt(design, mask, sim)

        # 3. Aerial image
        aerial = sim(mask_opc)

        # 4. Save and display images
        save_img(design,  "design.png", ttl="Design")
        save_img(mask_opc, "opc.png",    ttl=f"OPC ({opc_method})")
        save_img(aerial,   "aerial.png", cmap="inferno", ttl="Aerial")
        save_profile(aerial, "profile.png")

        for path, lab in zip(["design.png", "opc.png", "aerial.png", "profile.png"], self.views):
            lab.setPixmap(QPixmap(path).scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))

# ──────────────────────────────────────────────
# 8. main entry
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OPCDemo(); win.show()
    sys.exit(app.exec_())
