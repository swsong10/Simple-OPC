import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QComboBox, QLineEdit, QFileDialog)
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt

# ---- Simulation Parameters ----
WAVELENGTHS = {"EUV": 13.5e-3, "ArF": 193e-3}  # in micrometers
NAs = {"EUV": 0.33, "ArF": 0.85}  # nominal NA
GRID_SIZE = 512  # simulation grid
GRID_SPACING = 0.01  # micrometers per pixel (adjust for scale)

# ---- Pattern Generation ----
def generate_mask(pattern_type, size_nm, pitch_nm, mask_tone):
    mask = np.zeros((GRID_SIZE, GRID_SIZE))
    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
    scale = 1e-3 / GRID_SPACING  # convert nm -> pixel (nm to um)

    if pattern_type == "Via":
        half = int((size_nm * scale) / 2)
        mask[cy - half:cy + half, cx - half:cx + half] = 1
    elif pattern_type == "Line":
        pitch_pix = int(pitch_nm * scale)
        line_w = pitch_pix // 2  # 50% duty cycle
        for x in range(cx - 3 * pitch_pix, cx + 3 * pitch_pix, pitch_pix):
            mask[:, x - line_w // 2:x + line_w // 2] = 1

    # Mask tone inversion (1 = clear, 0 = opaque)
    if mask_tone == "Bright":
        mask = 1 - mask
    return mask

# ---- Rule-based OPC ----
def apply_opc(mask, pattern_type):
    opc = mask.copy()
    from scipy.ndimage import binary_dilation

    # Bias: dilate by 1 pixel (~10 nm equivalent depending on scaling)
    opc = binary_dilation(opc, iterations=1)

    # Add simple serifs/hammerheads for demonstration
    if pattern_type == "Via":
        cx, cy = np.array(opc.shape) // 2
        opc[cy-2:cy+2, cx-10:cx-8] = 1  # small serif left/right
        opc[cy-2:cy+2, cx+8:cx+10] = 1
        opc[cy-10:cy-8, cx-2:cx+2] = 1  # serif top/bottom
        opc[cy+8:cy+10, cx-2:cx+2] = 1
    elif pattern_type == "Line":
        opc[:4, :] = opc[-4:, :] = 1  # hammerheads at edges (simplified)
    return opc

# ---- Aerial Image Simulation (Abbe) ----
def simulate_aerial(mask, source="Conventional", wl=193e-3, NA=0.85):
    # FFT grid coordinates
    M = mask.shape[0]
    fft_mask = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
    fx = np.fft.fftfreq(M, d=GRID_SPACING)
    fy = np.fft.fftfreq(M, d=GRID_SPACING)
    FX, FY = np.meshgrid(fx, fy)
    pupil = (FX**2 + FY**2) <= (NA / wl)**2

    def coherent_image(angle_shift=(0,0)):
        field = fft_mask * pupil
        image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))
        return np.abs(image)**2

    # Source sampling
    sources = [(0,0)]
    if source == "Annular":
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        sources = [(0.7*np.cos(a),0.7*np.sin(a)) for a in angles]
    elif source == "Quadrupole":
        sources = [(0.7,0),( -0.7,0),(0,0.7),(0,-0.7)]

    aerial = np.zeros_like(mask, dtype=float)
    for s in sources:
        aerial += coherent_image(s)
    aerial /= np.max(aerial)
    return aerial

# ---- Visualization ----
def save_image(data, filename, cmap='gray'):
    plt.figure(figsize=(3,3))
    plt.axis('off')
    plt.imshow(data, cmap=cmap, origin='lower')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()

# ---- PyQt Application ----
class OPCDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPC & Aerial Image Demo")
        layout = QVBoxLayout()

        # Input selectors
        self.source_box = QComboBox(); self.source_box.addItems(["EUV","ArF"])
        self.pattern_box = QComboBox(); self.pattern_box.addItems(["Via","Line"])
        self.mask_tone_box = QComboBox(); self.mask_tone_box.addItems(["Bright","Dark"])
        self.illum_box = QComboBox(); self.illum_box.addItems(["Conventional","Annular","Quadrupole"])
        self.size_input = QLineEdit("100")  # nm
        self.pitch_input = QLineEdit("200")  # nm

        for lbl, widget in [("Source",self.source_box),
                            ("Pattern",self.pattern_box),
                            ("Mask Tone",self.mask_tone_box),
                            ("Illumination",self.illum_box),
                            ("Size (nm)",self.size_input),
                            ("Pitch (nm)",self.pitch_input)]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl)); row.addWidget(widget)
            layout.addLayout(row)

        run_btn = QPushButton("Run Simulation")
        run_btn.clicked.connect(self.run_simulation)
        layout.addWidget(run_btn)

        # Output images
        self.img_labels = [QLabel() for _ in range(3)]
        row = QHBoxLayout()
        for lbl in self.img_labels: row.addWidget(lbl)
        layout.addLayout(row)

        self.setLayout(layout)

    def run_simulation(self):
        source = self.source_box.currentText()
        pattern = self.pattern_box.currentText()
        tone = self.mask_tone_box.currentText()
        illum = self.illum_box.currentText()
        size = float(self.size_input.text())
        pitch = float(self.pitch_input.text())

        mask = generate_mask(pattern, size, pitch, tone)
        opc_mask = apply_opc(mask, pattern)
        aerial = simulate_aerial(opc_mask, illum, wl=WAVELENGTHS[source], NA=NAs[source])

        save_image(mask, "mask.png")
        save_image(opc_mask, "opc.png")
        save_image(aerial, "aerial.png", cmap='inferno')

        for path, lbl in zip(["mask.png","opc.png","aerial.png"], self.img_labels):
            lbl.setPixmap(QPixmap(path).scaled(256,256))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OPCDemo(); win.show()
    sys.exit(app.exec_())
