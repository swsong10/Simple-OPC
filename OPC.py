import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_closing, label
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
from pathlib import Path

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════

@dataclass(frozen=True)
class LithoConfig:
    """Immutable lithography configuration"""
    GRID: int = 256
    VIEW_PIXELS: int = 250
    WAVELENGTHS: dict = None
    NAS: dict = None
    SQRT2: float = np.sqrt(2.0)
    
    def __post_init__(self):
        if self.WAVELENGTHS is None:
            object.__setattr__(self, 'WAVELENGTHS', {"EUV": 13.5e-3, "ArF": 193e-3})
        if self.NAS is None:
            object.__setattr__(self, 'NAS', {"EUV": 0.33, "ArF": 0.85})

CONFIG = LithoConfig()

# ══════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════

def nm_to_pixels(nm: float, nm_per_px: float) -> int:
    """Convert nanometers to pixel count"""
    return max(1, int(round(nm / nm_per_px)))


def generate_diagonal_centers(n: int, pitch: float, cx: int, cy: int):
    """Generate diagonal via array center coordinates"""
    step = pitch / CONFIG.SQRT2
    offset = -(n // 2)
    
    for i in range(n):
        for j in range(n):
            u, v = offset + i, offset + j
            x = int(cx + round((u - v) * step))
            y = int(cy + round((u + v) * step))
            yield x, y


def safe_region_fill(array: np.ndarray, y_slice: slice, x_slice: slice, value: float = 1.0):
    """Safely fill array region with bounds checking"""
    h, w = array.shape
    y_start = max(0, y_slice.start or 0)
    y_stop = min(h, y_slice.stop or h)
    x_start = max(0, x_slice.start or 0)
    x_stop = min(w, x_slice.stop or w)
    
    if y_start < y_stop and x_start < x_stop:
        array[y_start:y_stop, x_start:x_stop] = value


def get_center_intensity(aerial: np.ndarray) -> float:
    """
    Get peak intensity in center region of the image
    For line patterns, find the brightest spot in center area
    """
    cy, cx = aerial.shape[0] // 2, aerial.shape[1] // 2
    # Use larger region (11x11) to ensure we catch features
    half_size = 5
    region = aerial[cy-half_size:cy+half_size+1, cx-half_size:cx+half_size+1]
    return float(np.max(region))

def calculate_ils(aerial: np.ndarray) -> float:
    """
    Calculate Image Log Slope (ILS) - measure of edge sharpness
    ILS = maximum slope of log(intensity) across center profile
    Higher ILS = sharper edges = better pattern fidelity
    """
    cy = aerial.shape[0] // 2
    profile = aerial[cy, :]
    profile = np.maximum(profile, 1e-10)
    log_profile = np.log10(profile)
    gradient = np.abs(np.gradient(log_profile))
    ils = np.max(gradient)
    return float(ils)

# ══════════════════════════════════════════════
# DESIGN PATTERN GENERATION
# ══════════════════════════════════════════════

class DesignPattern:
    """Base class for pattern generation"""
    
    @staticmethod
    def create(pattern_type: str, size_nm: float, pitch_nm: float, 
               tmpl_nm: float, nm_per_px: float) -> Tuple[np.ndarray, str]:
        """Factory method for pattern creation"""
        is_via = "Via" in pattern_type
        is_diagonal = pattern_type == "Diagonal Via"
        
        pattern_cls = ViaPattern if is_via else LinePattern
        return pattern_cls.generate(size_nm, pitch_nm, tmpl_nm, nm_per_px, is_diagonal)


class ViaPattern:
    """Via (contact hole) pattern generator"""
    
    @staticmethod
    def generate(size_nm: float, pitch_nm: float, tmpl_nm: float, 
                 nm_per_px: float, diagonal: bool = False) -> Tuple[np.ndarray, str]:
        design = np.zeros((CONFIG.GRID, CONFIG.GRID), dtype=float)
        cx = cy = CONFIG.GRID // 2
        
        size_px = nm_to_pixels(size_nm, nm_per_px)
        pitch_px = nm_to_pixels(pitch_nm, nm_per_px)
        tmpl_px = nm_to_pixels(tmpl_nm, nm_per_px)
        
        half_size = size_px // 2
        n_vias = max(1, (tmpl_px // pitch_px) | 1)
        
        if diagonal:
            centers = generate_diagonal_centers(n_vias, pitch_px, cx, cy)
        else:
            centers = (
                (cx + (i - n_vias//2) * pitch_px, cy + (j - n_vias//2) * pitch_px)
                for i in range(n_vias) for j in range(n_vias)
            )
        
        for x0, y0 in centers:
            safe_region_fill(design, 
                           slice(y0 - half_size, y0 + half_size),
                           slice(x0 - half_size, x0 + half_size))
        
        return design, "Via"


class LinePattern:
    """Line/Space pattern generator"""
    
    @staticmethod
    def generate(size_nm: float, pitch_nm: float, tmpl_nm: float, 
                 nm_per_px: float, *args) -> Tuple[np.ndarray, str]:
        design = np.zeros((CONFIG.GRID, CONFIG.GRID), dtype=float)
        cx = cy = CONFIG.GRID // 2
        
        line_width = nm_to_pixels(size_nm, nm_per_px)
        pitch_px = nm_to_pixels(pitch_nm, nm_per_px)
        tmpl_px = nm_to_pixels(tmpl_nm, nm_per_px)
        
        half_tmpl = tmpl_px // 2
        y_margin = int(0.10 * tmpl_px)
        y0 = cy - half_tmpl + y_margin
        y1 = cy + half_tmpl - y_margin
        
        for offset in range(0, half_tmpl + pitch_px, pitch_px):
            x_offsets = [0] if offset == 0 else [-offset, offset]
            
            for sx in x_offsets:
                x0 = cx + sx - line_width // 2
                x1 = x0 + line_width
                
                if 0 <= x0 < CONFIG.GRID and x1 <= CONFIG.GRID:
                    design[y0:y1, x0:x1] = 1.0
        
        return design, "Line"


# ══════════════════════════════════════════════
# OPC TECHNIQUES
# ══════════════════════════════════════════════

class OPCProcessor:
    """Optical Proximity Correction processor"""
    
    HAMMERHEAD_THICKNESS = 10
    SERIF_THICKNESS = 10
    
    @staticmethod
    def apply_tone_and_resist(mask: np.ndarray, tone: str, resist: str) -> np.ndarray:
        """Apply mask tone and resist type transformation"""
        result = (1 - mask) if tone == "Bright" else mask.copy()
        return (1 - result) if resist == "NTD" else result
    
    @classmethod
    def rule_based(cls, mask: np.ndarray, pattern_type: str) -> np.ndarray:
        """
        Rule-based OPC: comprehensive corrections
        
        Line patterns:
        - Line width bias (2px dilation)
        - Line-end extension (LESC - Line End Shortening Compensation)
        - Hammerheads at top/bottom
        - Corner serifs for better corner rounding
        - Inner serifs to prevent corner rounding loss
        
        Via patterns:
        - Standard bias + corner serifs
        """
        if pattern_type == "Line":
            return cls._rule_line_advanced(mask)
        else:
            return cls._rule_via_standard(mask)
    
    @classmethod
    def _rule_line_advanced(cls, mask: np.ndarray) -> np.ndarray:
        """Advanced rule-based OPC for lines"""
        # Start with 2-pixel bias for line width compensation
        opc = binary_dilation(mask.astype(bool), iterations=2).astype(float)
        
        # Find individual line features
        components, n_comp = label(mask.astype(bool))
        
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            line_width = x_max - x_min + 1
            line_height = y_max - y_min + 1
            
            # Line-End Shortening Compensation (extend vertically)
            lesc_extension = 8  # pixels to extend at each end
            safe_region_fill(opc, 
                           slice(max(0, y_min - lesc_extension), y_min),
                           slice(x_min, x_max + 1))
            safe_region_fill(opc,
                           slice(y_max + 1, min(CONFIG.GRID, y_max + 1 + lesc_extension)),
                           slice(x_min, x_max + 1))
            
            # Corner serifs (outer)
            serif_size = max(6, line_width // 3)
            corners = [
                (y_min - serif_size, x_min - serif_size),  # Top-left
                (y_min - serif_size, x_max + 1),           # Top-right
                (y_max + 1, x_min - serif_size),           # Bottom-left
                (y_max + 1, x_max + 1),                    # Bottom-right
            ]
            
            for y, x in corners:
                safe_region_fill(opc, 
                               slice(y, y + serif_size), 
                               slice(x, x + serif_size))
            
            # Inner serifs (prevent corner pullback)
            inner_serif = 4
            safe_region_fill(opc,
                           slice(y_min, y_min + inner_serif),
                           slice(x_min - inner_serif, x_min))
            safe_region_fill(opc,
                           slice(y_min, y_min + inner_serif),
                           slice(x_max + 1, x_max + 1 + inner_serif))
            safe_region_fill(opc,
                           slice(y_max + 1 - inner_serif, y_max + 1),
                           slice(x_min - inner_serif, x_min))
            safe_region_fill(opc,
                           slice(y_max + 1 - inner_serif, y_max + 1),
                           slice(x_max + 1, x_max + 1 + inner_serif))
        
        # Global hammerheads at top and bottom
        T = cls.HAMMERHEAD_THICKNESS
        opc[:T, :] = 1.0
        opc[-T:, :] = 1.0
        
        return opc
    
    @classmethod
    def _rule_via_standard(cls, mask: np.ndarray) -> np.ndarray:
        """Standard rule-based OPC for vias"""
        opc = binary_dilation(mask.astype(bool), iterations=1).astype(float)
        cls._apply_via_serifs(mask, opc)
        return opc
    
    @classmethod
    def _apply_hammerhead(cls, opc: np.ndarray):
        """Apply hammerhead at line ends"""
        T = cls.HAMMERHEAD_THICKNESS
        opc[:T, :] = 1.0
        opc[-T:, :] = 1.0
    
    @classmethod
    def _apply_line_serifs(cls, mask: np.ndarray, opc: np.ndarray):
        """Apply corner serifs for line patterns"""
        components, n_comp = label(mask.astype(bool))
        T = cls.SERIF_THICKNESS
        
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            corners = [
                (y_min - T, x_min - T),
                (y_max + 1, x_min - T),
                (y_min - T, x_max + 1),
                (y_max + 1, x_max + 1),
            ]
            
            for y, x in corners:
                safe_region_fill(opc, slice(y, y + T), slice(x, x + T))
    
    @classmethod
    def _apply_via_serifs(cls, mask: np.ndarray, opc: np.ndarray):
        """Apply corner serifs for via patterns"""
        components, n_comp = label(mask.astype(bool))
        T_half = cls.SERIF_THICKNESS // 2
        
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            
            corners = [
                (y_min - T_half, x_min - T_half),
                (y_max + 1, x_min - T_half),
                (y_min - T_half, x_max + 1),
                (y_max + 1, x_max + 1),
            ]
            
            for y, x in corners:
                safe_region_fill(opc, slice(y, y + T_half), slice(x, x + T_half))
    
    @staticmethod
    def curvilinear(mask: np.ndarray, pattern_type: str, size_px: int) -> np.ndarray:
        """
        Curvilinear OPC - smooth rounded features
        
        Line patterns: 
        - Rounded corners with aggressive smoothing
        - Line end rounding
        - Overall shape smoothing for better optical fidelity
        
        Via patterns:
        - Circular geometry
        """
        if pattern_type == "Via":
            return OPCProcessor._curvi_via(mask, size_px)
        else:
            return OPCProcessor._curvi_line_advanced(mask, size_px)
    
    @staticmethod
    def _curvi_line_advanced(mask: np.ndarray, size_px: int) -> np.ndarray:
        """
        Advanced curvilinear for lines with proper rounding
        
        Strategy:
        1. Start with slight bias
        2. Apply aggressive morphological smoothing
        3. Round line ends with circular caps
        4. Smooth corners iteratively
        """
        # Start with 1px bias
        opc = binary_dilation(mask.astype(bool), iterations=1).astype(bool)
        
        # Multiple rounds of closing with different structuring elements
        # This creates rounded corners
        struct_3x3 = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=bool)
        
        struct_5x5 = np.array([[0, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0],
                               [1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 0],
                               [0, 0, 1, 0, 0]], dtype=bool)
        
        # Apply smoothing
        opc = binary_closing(opc, structure=struct_3x3, iterations=2)
        opc = binary_closing(opc, structure=struct_5x5, iterations=1)
        
        # Add rounded caps at line ends
        components, n_comp = label(mask.astype(bool))
        Y, X = np.ogrid[:CONFIG.GRID, :CONFIG.GRID]
        
        result = opc.astype(float)
        
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cx = (x_min + x_max) // 2
            line_width = x_max - x_min + 1
            
            # Circular caps at top and bottom
            cap_radius = int(line_width * 0.7)
            
            # Top cap
            cap_y_top = y_min
            result[(X - cx)**2 + (Y - cap_y_top)**2 <= cap_radius**2] = 1.0
            
            # Bottom cap
            cap_y_bottom = y_max
            result[(X - cx)**2 + (Y - cap_y_bottom)**2 <= cap_radius**2] = 1.0
        
        return result
    
    @staticmethod
    def _curvi_via(mask: np.ndarray, size_px: int) -> np.ndarray:
        """Circular vias for better fidelity"""
        radius = int(size_px / 1.3)
        components, n_comp = label(mask.astype(bool))
        Y, X = np.ogrid[:CONFIG.GRID, :CONFIG.GRID]
        result = np.zeros_like(mask, dtype=float)
        
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            
            cx = int(xs.mean())
            cy = int(ys.mean())
            result[(X - cx)**2 + (Y - cy)**2 <= radius**2] = 1.0
        
        return result
    
    @staticmethod
    def _curvi_line(mask: np.ndarray) -> np.ndarray:
        """Rounded line edges"""
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        return binary_closing(mask.astype(bool), structure=structure).astype(float)
    
    @staticmethod
    def sub_resolution_assist(mask: np.ndarray, pattern_type: str, 
                              pitch_px: int, size_px: int) -> np.ndarray:
        """
        Advanced Sub-Resolution Assist Features (SRAF)
        
        Strategy: Start with curvilinear base (rounded features) 
        then add optimized assist features
        
        This combines the benefits of:
        - Curvilinear OPC (smooth edges, rounded corners)
        - SRAF (constructive interference for center intensity boost)
        """
        if pattern_type == "Line":
            return OPCProcessor._sraf_line_optimized(mask, pitch_px, size_px)
        else:
            return OPCProcessor._sraf_via_optimized(mask, pitch_px, size_px)
    
    @staticmethod
    def _sraf_line_optimized(mask: np.ndarray, pitch_px: int, size_px: int) -> np.ndarray:
        """
        Optimized SRAF for line patterns with curvilinear base
        
        Process:
        1. Start with curvilinear OPC (rounded features)
        2. Add sub-resolution assist bars at optimal distance
        3. Assist features enhance main feature without printing themselves
        """
        # Start with curvilinear base for smooth main features
        opc = OPCProcessor._curvi_line_advanced(mask, size_px)
        
        # SRAF parameters
        sraf_offset = int(0.45 * pitch_px)
        sraf_width = max(2, size_px // 5)
        sraf_transmittance = 0.65
        
        # Find main line regions from original mask
        center_line = mask[CONFIG.GRID // 2]
        main_cols = np.where(center_line == 1)[0]
        
        if main_cols.size > 0:
            groups = np.split(main_cols, np.where(np.diff(main_cols) != 1)[0] + 1)
            
            for group in groups:
                cx = int(np.mean(group))
                line_ys = np.where(mask[:, cx] == 1)[0]
                if line_ys.size == 0:
                    continue
                
                y_min, y_max = line_ys.min(), line_ys.max()
                
                # Place SRAFs on both sides
                for dx in (-sraf_offset, sraf_offset):
                    x0 = cx + dx - sraf_width // 2
                    x1 = x0 + sraf_width
                    
                    if 0 <= x0 < CONFIG.GRID and x1 <= CONFIG.GRID:
                        # Only add SRAF where there's no main feature
                        region = opc[y_min:y_max+1, x0:x1]
                        mask_empty = region < 0.5
                        opc[y_min:y_max+1, x0:x1] = np.where(
                            mask_empty, sraf_transmittance, region
                        )
        
        return opc
    
    @staticmethod
    def _sraf_via_optimized(mask: np.ndarray, pitch_px: int, size_px: int) -> np.ndarray:
        """
        Optimized SRAF for via patterns with curvilinear base
        
        Process:
        1. Start with circular main vias (curvilinear)
        2. Add 8-directional assist features for isotropic enhancement
        """
        Y, X = np.ogrid[:CONFIG.GRID, :CONFIG.GRID]
        opc = np.zeros_like(mask, dtype=float)
        
        # Enhanced main via (curvilinear base)
        radius_main = int(size_px / 1.2)
        components, n_comp = label(mask.astype(bool))
        
        via_centers = []
        for idx in range(1, n_comp + 1):
            ys, xs = np.where(components == idx)
            if xs.size == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            via_centers.append((cx, cy))
            opc[(X - cx)**2 + (Y - cy)**2 <= radius_main**2] = 1.0
        
        # SRAF parameters
        assist_offset = int(0.42 * pitch_px)
        radius_assist = max(2, size_px // 4)
        assist_transmittance = 0.65
        
        # Add assist features around each via
        for cx, cy in via_centers:
            diag_offset = int(assist_offset / np.sqrt(2))
            
            for dx, dy in [
                (-assist_offset, 0), (assist_offset, 0),
                (0, -assist_offset), (0, assist_offset),
                (-diag_offset, -diag_offset), (diag_offset, -diag_offset),
                (-diag_offset, diag_offset), (diag_offset, diag_offset),
            ]:
                assist_x = cx + dx
                assist_y = cy + dy
                
                if 0 <= assist_x < CONFIG.GRID and 0 <= assist_y < CONFIG.GRID:
                    mask_assist = (X - assist_x)**2 + (Y - assist_y)**2 <= radius_assist**2
                    opc[mask_assist] = np.maximum(opc[mask_assist], assist_transmittance)
        
        return opc


# ══════════════════════════════════════════════
# OPTICAL SIMULATOR
# ══════════════════════════════════════════════

class OpticsSimulator:
    """Lithography optics simulation"""
    
    @staticmethod
    def create_pupil(FX: np.ndarray, FY: np.ndarray, na: float, 
                     wavelength: float, illumination: str) -> np.ndarray:
        """Create pupil function for different illumination modes"""
        k_max = na / wavelength
        R = np.hypot(FX, FY)
        
        if illumination == "Conventional":
            return (R <= k_max).astype(float)
        elif illumination == "Annular":
            return ((R <= k_max) & (R >= 0.6 * k_max)).astype(float)
        elif illumination == "Quadrupole":
            pupil = np.zeros_like(R, dtype=bool)
            pole_radius = 0.15 * k_max
            pole_center = 0.7 * k_max
            
            for cx, cy in [(pole_center, 0), (-pole_center, 0), 
                          (0, pole_center), (0, -pole_center)]:
                pupil |= ((FX - cx)**2 + (FY - cy)**2 <= pole_radius**2)
            
            return pupil.astype(float)
        
        return (R <= k_max).astype(float)
    
    @staticmethod
    def build_simulator(wavelength: float, na: float, illumination: str, 
                       um_per_pixel: float) -> Callable:
        """Build aerial image simulator with Hopkins approximation"""
        freq = np.fft.fftshift(np.fft.fftfreq(CONFIG.GRID, d=um_per_pixel))
        FX, FY = np.meshgrid(freq, freq)
        pupil = OpticsSimulator.create_pupil(FX, FY, na, wavelength, illumination)
        
        def simulate(mask: np.ndarray) -> np.ndarray:
            """Compute aerial image from mask"""
            mask_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
            filtered = mask_fft * pupil
            image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(filtered)))
            return np.abs(image) ** 2
        
        return simulate


# ══════════════════════════════════════════════
# IMAGE EXPORT
# ══════════════════════════════════════════════

class ImageExporter:
    """Handle image and profile exports"""
    
    @staticmethod
    def save_pattern(array: np.ndarray, path: str, title: str = "", 
                    cmap: str = "gray", vmax: Optional[float] = None):
        """Save 2D pattern image"""
        plt.figure(figsize=(3, 3))
        if vmax is not None:
            plt.imshow(array, cmap=cmap, origin="lower", interpolation='nearest', 
                      vmin=0, vmax=vmax)
        else:
            plt.imshow(array, cmap=cmap, origin="lower", interpolation='nearest')
        plt.title(title, fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=90, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_profile(intensity: np.ndarray, path: str, center_val: float, 
                    ils_val: float, ref_max: Optional[float] = None):
        """Save intensity profile with peak intensity and ILS metrics"""
        mid_row = intensity.shape[0] // 2
        profile = intensity[mid_row, :]
        
        y_max = ref_max if ref_max is not None else max(1.0, profile.max() * 1.1)
        
        plt.figure(figsize=(3, 3))
        plt.plot(profile, linewidth=2, color='#2196F3')
        plt.axhline(y=center_val, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Peak: {center_val:.3f}')
        plt.ylim(0, y_max)
        plt.xlabel("X Position (px)", fontsize=9)
        plt.ylabel("Intensity", fontsize=9)
        plt.title(f"Profile (ILS={ils_val:.3f})", fontsize=10)
        plt.legend(fontsize=8, loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=90, bbox_inches='tight')
        plt.close()


# ══════════════════════════════════════════════
# GUI APPLICATION
# ══════════════════════════════════════════════

class OPCDemoApp(QWidget):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.reference_intensity: Optional[float] = None
        self.baseline_center: Optional[float] = None
        self.baseline_ils: Optional[float] = None  # Store baseline ILS
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("OPC Lithography Simulator")
        self.resize(1140, 520)
        
        main_layout = QVBoxLayout(self)
        
        top_bar = self.create_top_bar()
        main_layout.addLayout(top_bar)
        
        controls = self.create_controls()
        main_layout.addLayout(controls)
        
        run_btn = QPushButton("▶ Run Simulation")
        run_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        run_btn.clicked.connect(self.run_simulation)
        main_layout.addWidget(run_btn)
        
        image_layout = self.create_image_display()
        main_layout.addLayout(image_layout)
    
    def create_top_bar(self) -> QHBoxLayout:
        """Create top bar with intensity reset button"""
        layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset Intensity Reference")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                padding: 6px 12px;
                background-color: #FF9800;
                color: white;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_intensity_reference)
        
        self.status_label = QLabel("No reference set - First simulation will set baseline")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 6px;
                background-color: #E3F2FD;
                border-radius: 4px;
                color: #1976D2;
            }
        """)
        
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        return layout
    
    def reset_intensity_reference(self):
        """Reset the intensity reference baseline"""
        self.reference_intensity = None
        self.baseline_center = None
        self.baseline_ils = None
        self.status_label.setText("✓ Reference cleared - Next simulation will set new baseline")
        self.status_label.setStyleSheet("""
            QLabel {
                padding: 6px;
                background-color: #FFF9C4;
                border-radius: 4px;
                color: #F57F17;
                font-weight: bold;
            }
        """)
        
        for label in self.image_labels:
            label.clear()
            label.setText("Run simulation\nto see results")
            label.setAlignment(Qt.AlignCenter)
    
    def update_status(self, center_intensity: float, ils: float, 
                     improvement: Optional[float] = None, ils_improvement: Optional[float] = None):
        """Update status label with intensity and ILS metrics"""
        if improvement is None:
            msg = f"✓ Baseline: Peak={center_intensity:.4f}, ILS={ils:.3f}"
            color = "#C8E6C9"
            text_color = "#2E7D32"
        else:
            sign_int = "+" if improvement >= 0 else ""
            sign_ils = "+" if ils_improvement >= 0 else ""
            msg = (f"Peak: {center_intensity:.4f} ({sign_int}{improvement:.1f}%) | "
                   f"ILS: {ils:.3f} ({sign_ils}{ils_improvement:.1f}%)")
            
            # Color based on combined performance
            if improvement >= 3 and ils_improvement >= 3:
                color = "#C8E6C9"  # Both improved significantly
                text_color = "#2E7D32"
            elif improvement >= 0 and ils_improvement >= 0:
                color = "#FFF9C4"  # Both improved slightly
                text_color = "#F57F17"
            else:
                color = "#FFCDD2"  # At least one decreased
                text_color = "#C62828"
        
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                padding: 6px;
                background-color: {color};
                border-radius: 4px;
                color: {text_color};
                font-weight: bold;
            }}
        """)
    
    def create_controls(self) -> QVBoxLayout:
        """Create control panel widgets"""
        layout = QVBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["EUV", "ArF"])
        
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Orthogonal Via", "Diagonal Via", "Line"])
        
        self.tone_combo = QComboBox()
        self.tone_combo.addItems(["Dark", "Bright"])
        
        self.resist_combo = QComboBox()
        self.resist_combo.addItems(["PTD", "NTD"])
        
        self.illum_combo = QComboBox()
        self.illum_combo.addItems(["Conventional", "Annular", "Quadrupole"])
        
        self.opc_combo = QComboBox()
        self.opc_combo.addItems(["No OPC", "Rule", "Curvilinear", "SRAF"])
        
        self.size_edit = QLineEdit("30")
        self.pitch_edit = QLineEdit("150")
        self.template_edit = QLineEdit("500")
        
        controls = [
            ("Light Source:", self.source_combo),
            ("Pattern Type:", self.pattern_combo),
            ("Mask Tone:", self.tone_combo),
            ("Resist Type:", self.resist_combo),
            ("Illumination:", self.illum_combo),
            ("OPC Method:", self.opc_combo),
            ("Feature Size (nm):", self.size_edit),
            ("Pitch (nm):", self.pitch_edit),
            ("Template Size (nm):", self.template_edit),
        ]
        
        for label_text, widget in controls:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setMinimumWidth(150)
            row.addWidget(label)
            row.addWidget(widget)
            layout.addLayout(row)
        
        return layout
    
    def create_image_display(self) -> QHBoxLayout:
        """Create image display area"""
        layout = QHBoxLayout()
        self.image_labels = []
        
        for title in ["Design", "OPC Mask", "Aerial Image", "Profile"]:
            container = QVBoxLayout()
            label = QLabel()
            label.setFixedSize(270, 270)
            label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
            label.setAlignment(Qt.AlignCenter)
            
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold;")
            
            container.addWidget(title_label)
            container.addWidget(label)
            layout.addLayout(container)
            
            self.image_labels.append(label)
        
        return layout
    
    def run_simulation(self):
        """Execute lithography simulation with intensity tracking"""
        try:
            source = self.source_combo.currentText()
            wavelength = CONFIG.WAVELENGTHS[source]
            na = CONFIG.NAS[source]
            
            size_nm = float(self.size_edit.text())
            pitch_nm = float(self.pitch_edit.text())
            template_nm = float(self.template_edit.text())
            
            if size_nm <= 0 or pitch_nm <= 0 or template_nm <= 0:
                raise ValueError("All dimensions must be positive")
            
            nm_per_pixel = template_nm / CONFIG.VIEW_PIXELS
            um_per_pixel = nm_per_pixel * 1e-3
            
            pattern_type = self.pattern_combo.currentText()
            design, pattern_class = DesignPattern.create(
                pattern_type, size_nm, pitch_nm, template_nm, nm_per_pixel
            )
            
            opc_method = self.opc_combo.currentText()
            
            if opc_method == "No OPC":
                mask_opc = design.copy()
            elif opc_method == "Rule":
                mask_opc = OPCProcessor.rule_based(design, pattern_class)
            elif opc_method == "Curvilinear":
                size_px = nm_to_pixels(size_nm, nm_per_pixel)
                mask_opc = OPCProcessor.curvilinear(design, pattern_class, size_px)
            else:
                pitch_px = nm_to_pixels(pitch_nm, nm_per_pixel)
                size_px = nm_to_pixels(size_nm, nm_per_pixel)
                mask_opc = OPCProcessor.sub_resolution_assist(
                    design, pattern_class, pitch_px, size_px
                )
            
            tone = self.tone_combo.currentText()
            resist = self.resist_combo.currentText()
            mask_final = OPCProcessor.apply_tone_and_resist(mask_opc, tone, resist)
            
            illumination = self.illum_combo.currentText()
            simulator = OpticsSimulator.build_simulator(
                wavelength, na, illumination, um_per_pixel
            )
            aerial_raw = simulator(mask_final)
            
            current_max = aerial_raw.max()
            
            if self.reference_intensity is None:
                # First simulation: set reference
                self.reference_intensity = current_max
                self.baseline_center = get_center_intensity(aerial_raw / self.reference_intensity)
                self.baseline_ils = calculate_ils(aerial_raw / self.reference_intensity)
                
                aerial_normalized = aerial_raw / self.reference_intensity
                center_intensity = self.baseline_center
                ils = self.baseline_ils
                
                self.update_status(center_intensity, ils, improvement=None, ils_improvement=None)
            else:
                # Subsequent simulations: use saved reference
                aerial_normalized = aerial_raw / self.reference_intensity
                center_intensity = get_center_intensity(aerial_normalized)
                ils = calculate_ils(aerial_normalized)
                
                # Calculate improvements
                improvement = ((center_intensity / self.baseline_center) - 1.0) * 100
                ils_improvement = ((ils / self.baseline_ils) - 1.0) * 100
                
                self.update_status(center_intensity, ils, improvement, ils_improvement)
            
            display_max = max(1.2, aerial_normalized.max() * 1.1)
            
            output_dir = Path(".")
            paths = [
                output_dir / "design.png",
                output_dir / "opc_mask.png",
                output_dir / "aerial.png",
                output_dir / "profile.png"
            ]
            
            ImageExporter.save_pattern(design, str(paths[0]), "Design Pattern")
            ImageExporter.save_pattern(mask_opc, str(paths[1]), f"OPC: {opc_method}")
            ImageExporter.save_pattern(
                aerial_normalized, str(paths[2]), 
                f"Aerial (Peak: {center_intensity:.3f})", 
                cmap="inferno", vmax=display_max
            )
            ImageExporter.save_profile(
                aerial_normalized, str(paths[3]), 
                center_intensity, ils, ref_max=display_max
            )
            
            for label, path in zip(self.image_labels, paths):
                pixmap = QPixmap(str(path))
                label.setPixmap(pixmap.scaled(
                    260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
        
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"Error: {str(e)}")


# ══════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════

def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = OPCDemoApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()