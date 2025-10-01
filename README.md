# OPC Lithography Simulator

Advanced Optical Proximity Correction (OPC) simulator for semiconductor lithography pattern analysis.

## Features

### Pattern Types
- **Orthogonal Via**: Square contact hole arrays
- **Diagonal Via**: 45° rotated via arrays
- **Line/Space**: Vertical line patterns

### OPC Methods
1. **No OPC**: Baseline pattern without correction
2. **Rule-based OPC**: 
   - Line width bias (2px dilation)
   - Line-End Shortening Compensation (LESC)
   - Corner serifs and hammerheads
   - Inner serifs for corner protection
3. **Curvilinear OPC**:
   - Rounded corners and edges
   - Circular end caps for lines
   - Morphological smoothing
4. **SRAF (Sub-Resolution Assist Features)**:
   - Curvilinear base + assist features
   - Optimized placement at 0.45× pitch
   - Attenuated transmission (65%)

### Lithography Sources
- **EUV**: 13.5 nm wavelength, NA=0.33
- **ArF**: 193 nm wavelength, NA=0.85

### Illumination Modes
- **Conventional**: Circular pupil
- **Annular**: Ring-shaped illumination (0.6-1.0 σ)
- **Quadrupole**: Four-pole off-axis illumination

### Metrics
- **Peak Intensity**: Maximum intensity in center region
- **ILS (Image Log Slope)**: Edge sharpness metric
  - Higher ILS = sharper edges = better fidelity

## Installation

### Requirements
- Python 3.7+
- pip package manager

### Setup
```bash
# Clone or download the repository
git clone <repository-url>
cd opc-simulator

# Install dependencies
pip install -r requirements.txt

# Run the simulator
python opc_simulator.py
```

## Usage

### Basic Workflow
1. **Set Parameters**:
   - Select light source (EUV/ArF)
   - Choose pattern type
   - Set mask tone (Dark/Bright) and resist type (PTD/NTD)
   - Configure illumination mode
   - Enter feature size, pitch, and template size

2. **First Simulation** (Baseline):
   - Select "No OPC"
   - Click "Run Simulation"
   - This sets the reference intensity for comparisons

3. **Compare OPC Methods**:
   - Change OPC method
   - Run simulation again
   - View improvement percentages for Peak and ILS

4. **Reset Reference**:
   - Click "🔄 Reset Intensity Reference" to start fresh
   - Useful when changing major parameters (source, pattern, etc.)

### Example Parameters
**EUV Line Pattern:**
- Light Source: EUV
- Pattern: Line
- Mask Tone: Dark
- Resist: PTD
- Illumination: Quadrupole
- Feature Size: 30 nm
- Pitch: 150 nm
- Template: 500 nm

**ArF Via Pattern:**
- Light Source: ArF
- Pattern: Orthogonal Via
- Mask Tone: Bright
- Resist: NTD
- Illumination: Annular
- Feature Size: 80 nm
- Pitch: 200 nm
- Template: 600 nm

## Output

### Displays (Left to Right)
1. **Design**: Original pattern geometry
2. **OPC Mask**: Pattern after OPC corrections
3. **Aerial Image**: Simulated wafer intensity (Hopkins model)
4. **Profile**: Center-line intensity profile with ILS

### Status Bar
Shows real-time comparison vs baseline:
- 🟢 Green: Both Peak and ILS improved significantly (≥3%)
- 🟡 Yellow: Both improved moderately (<3%)
- 🔴 Red: At least one metric decreased

### Saved Files
Images are automatically saved to current directory:
- `design.png`
- `opc_mask.png`
- `aerial.png`
- `profile.png`

## Technical Details

### Optical Model
- **Hopkins Approximation**: Partially coherent imaging
- **Fourier Optics**: FFT-based simulation
- **Grid Resolution**: 256×256 pixels
- **Pupil Function**: Frequency-domain filtering

### OPC Algorithms

**Rule-based:**
- Deterministic geometric corrections
- Feature-specific bias and serifs
- Fast computation, proven reliability

**Curvilinear:**
- Smooth, rounded features
- Better optical fidelity
- Reduced corner rounding effects

**SRAF:**
- Sub-resolution assist features
- Constructive interference enhancement
- Combines curvilinear base with strategic assists

### Performance Tips
- Start with "No OPC" to establish baseline
- Use Quadrupole illumination for dense patterns
- Larger pitch allows more aggressive OPC
- SRAF works best with pitch > 4× feature size

## Troubleshooting

**Issue**: "No OPC gives low intensity (~0.77)"
- **Solution**: This is normal for line patterns where center falls between features. Use Reset button before comparing different pattern types.

**Issue**: "Illumination change drastically affects intensity"
- **Solution**: This is expected. Click Reset to establish new baseline for each illumination mode.

**Issue**: "OPC shows negative improvement"
- **Solution**: Some OPC methods prioritize edge sharpness (ILS) over peak intensity, or vice versa. Check both metrics.

## Theory Background

### Why OPC?
Optical diffraction causes features to print differently than designed. OPC pre-distorts the mask to compensate for:
- Line-end shortening
- Corner rounding  
- Proximity effects between features
- Optical blur

### ILS Explained
Image Log Slope measures how quickly intensity transitions from bright to dark:
```
ILS = max|d(log₁₀(I))/dx|
```
- Higher ILS → steeper edges → better CD control
- Critical for process window and dose latitude

### SRAF Strategy
Sub-resolution features placed strategically:
- Too close: They print (bad)
- Too far: No effect (useless)
- Optimal (~0.4-0.5× pitch): Enhance without printing

## License

MIT License - Feel free to use and modify

## Author

Created with assistance from Claude (Anthropic)

## Version

v1.0.0 - Initial release with Rule/Curvilinear/SRAF OPC methods
