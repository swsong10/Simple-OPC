import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLineEdit,
)
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, rotate

# ---------------------------
# GLOBAL SIMULATION CONSTANTS
# ---------------------------
WAVELENGTHS = {"EUV": 13.5e-3, "ArF": 193e-3}  # micrometres
NAs = {"EUV": 0.33, "ArF": 0.85}
GRID_SIZE = 512                       # square simulation grid
GRID_SPACING = 0.01                   # micrometres per pixel (=> 10 nm)

# ---------------------------
# HELPER CONVERSIONS
# ---------------------------

def nm_to_pix(length_nm: float) -> int:
    """Convert nanometres to nearest pixel index units."""
    return int(round(length_nm * 1e-3 / GRID_SPACING))


# ---------------------------
# PATTERN GENERATION
# ---------------------------

def generate_mask(
    gui_pattern: str,
    size_nm: float,
    pitch_nm: float,
    mask_tone: str,
    template_nm: float,
):
    """Return binary numpy array mask (1 = clear, 0 = opaque)."""
    pattern_type = "Via" if "Via" in gui_pattern else "Line"
    diagonal = gui_pattern == "Diagonal Via"

    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2

    size_px = nm_to_pix(size_nm)
    pitch_px = nm_to_pix(pitch_nm)
    template_px = nm_to_pix(template_nm)

    if pattern_type == "Via":
        # centre-aligned dense array filling template window
        num = max(1, (template_px // pitch_px) | 1)  # odd for symmetry
        half_via = size_px // 2
        offset0 = -(num // 2) * pitch_px

        for i in range(num):
            for j in range(num):
                x = cx + offset0 + i * pitch_px
                y = cy + offset0 + j * pitch_px
                mask[y - half_via : y + half_via, x - half_via : x + half_via] = 1

        if diagonal:
            mask = rotate(mask, 45, reshape=False, order=1, mode="constant", cval=0)

    else:  # Line pattern
        line_w_px = size_px  # size_nm is line width
        start_x = cx - template_px // 2
        end_x = cx + template_px // 2
        x = start_x + line_w_px // 2
        while x < end_x:
            mask[:, x - line_w_px // 2 : x + (line_w_px + 1) // 2] = 1
            x += pitch_px

    # invert for bright-field masks (background clear, features opaque)
    if mask_tone == "Bright":
        mask = 1 - mask
    return mask.astype(float), pattern_type  # return pattern_type for OPC use


# ---------------------------
# RULE-BASED OPC
# ---------------------------

def apply_opc(mask: np.ndarray, pattern_type: str) -> np.ndarray:
    """Very simple rule-based OPC: global bias + serifs/hammerheads."""
    opc = binary_dilation(mask, iterations=1).astype(float)  # bias ≈ +10 nm

    cy, cx = np.array(opc.shape) // 2
    if pattern_type == "Via":
        opc[cy - 3 : cy - 1, cx - 3 : cx - 1] = 1
        opc[cy - 3 : cy - 1, cx + 1 : cx + 3] = 1
        opc[cy + 1 : cy + 3, cx - 3 : cx - 1] = 1
        opc[cy + 1 : cy + 3, cx + 1 : cx + 3] = 1
    else:  # Line
        opc[:3, :] = 1
        opc[-3:, :] = 1
    return opc


# ---------------------------
# AERIAL IMAGE SIMULATION
# ---------------------------

def simulate_aerial(mask: np.ndarray, source: str, wl: float, NA: float) -> np.ndarray:
    """FFT-based incoherent sum of coherent images for given source."""
    M = mask.shape[0]
    fx = np.fft.fftfreq(M, d=GRID_SPACING)
    fy = np.fft.fftfreq(M, d=GRID_SPACING)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    pupil = (FX ** 2 + FY ** 2) <= (NA / wl) ** 2

    mask_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))

    def coherent_intensity():
        field = mask_ft * pupil
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
        return np.abs(img) ** 2

    if source == "Conventional":
        samples = [None]
    elif source == "Annular":
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        samples = angles  # placeholder, we ignore actual shift in this simple demo
    elif source == "Quadrupole":
        samples = range(4)
    else:
        samples = [None]

    intensity = np.zeros_like(mask, dtype=float)
    for _ in samples:
        intensity += coherent_intensity()
    intensity /= intensity.max()
    return intensity


# ---------------------------
# IMAGE SAVE UTILITY
# ---------------------------

def save_array_as_png(arr: np.ndarray, path: str, cmap: str = "gray"):
    plt.figure(figsize=(3, 3))
    plt.imshow(arr, cmap=cmap, origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------
# GUI APPLICATION
# ---------------------------

class OPCDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPC & Aerial Image Demo (Dense Arrays)")
        layout = QVBoxLayout(self)

        self.src_box = QComboBox(); self.src_box.addItems(["EUV", "ArF"])
        self.ptn_box = QComboBox(); self.ptn_box.addItems(["Orthogonal Via", "Diagonal Via", "Line"])
        self.tone_box = QComboBox(); self.tone_box.addItems(["Bright", "Dark"])
        self.illum_box = QComboBox(); self.illum_box.addItems(["Conventional", "Annular", "Quadrupole"])

        self.size_edit = QLineEdit("100")   # nm (via size or line width)
        self.pitch_edit = QLineEdit("200")  # nm
        self.temp_edit = QLineEdit("500")   # nm template window size

        def row(label, widget):
            h = QHBoxLayout(); h.addWidget(QLabel(label)); h.addWidget(widget); return h

        for lbl, w in [
            ("Source", self.src_box),
            ("Pattern", self.ptn_box),
            ("Mask Tone", self.tone_box),
            ("Illumination", self.illum_box),
            ("Feature size [nm]", self.size_edit),
            ("Pitch [nm]", self.pitch_edit),
            ("Template size [nm]", self.temp_edit),
        ]:
            layout.addLayout(row(lbl, w))

        run_btn = QPushButton("Run Simulation"); run_btn.clicked.connect(self.run)
        layout.addWidget(run_btn)

        self.labels = [QLabel() for _ in range(3)]
        img_row = QHBoxLayout()
        for l in self.labels: img_row.addWidget(l)
        layout.addLayout(img_row)

    def run(self):
        src = self.src_box.currentText()
        gui_ptype = self.ptn_box.currentText()
        tone = self.tone_box.currentText()
        illum = self.illum_box.currentText()
        size_nm = float(self.size_edit.text())
        pitch_nm = float(self.pitch_edit.text())
        tmpl_nm = float(self.temp_edit.text())

        mask, base_ptype = generate_mask(gui_ptype, size_nm, pitch_nm, tone, tmpl_nm)
        opc = apply_opc(mask, base_ptype)
        aerial = simulate_aerial(opc, illum, WAVELENGTHS[src], NAs[src])

        save_array_as_png(mask, "mask.png")
        save_array_as_png(opc, "opc.png")
        save_array_as_png(aerial, "aerial.png", cmap="inferno")

        for path, lbl in zip(["mask.png", "opc.png", "aerial.png"], self.labels):
            lbl.setPixmap(QPixmap(path).scaled(256, 256))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OPCDemo(); win.show()
    sys.exit(app.exec_())
