# AI Photo Editor Web (Advanced)

A simple, offline-capable photo editor built with **Flask (Python)** and **OpenCV + NumPy + Matplotlib**.  
Now includes an **Advanced Processing** module for intensity transforms and histogram operations.

## ✨ Features
### Basic
- **Blur**: Mean (Box), Gaussian, Median, **Bilateral (new)**
- **Sharpen**: Unsharp Mask, Laplacian Sharpen
- **Edge Detection**: Sobel, **Prewitt (new)**, Laplacian, Canny, Binary threshold
- Adjustable parameters: `kernel_size`, `sigma`, `amount`, `alpha`, `threshold`, `threshold1`, `threshold2`, and bilateral (`diameter`, `sigma_color`, `sigma_space`)
- Live preview while tweaking parameters
- Before/After side-by-side
- Download processed image
- JSON metadata saved next to each output

### Advanced
- **Intensity Transformations**
  - Negative (invert)
  - Log transform (with constant `c`)
  - Gamma correction (with `gamma` and `c`)
  - Piecewise-linear transform (`r1/s1/r2/s2` for contrast stretching)
- **Histogram Equalization**
  - Global HE
  - CLAHE (with `clip_limit` and `tile_grid_size`)
- Optional **histogram visualization** (before/after) generated with Matplotlib

### Presets
- **Brighten & Sharpen** (Unsharp defaults)
- **Soft skin smoothing** (Median blur)
- **Edge boost (low-contrast)** (Canny thresholds 50/150)
- **Auto Contrast Boost** (Histogram Equalization)
- **Low-light Enhancement** (Gamma < 1.0)

## 📂 Project Structure
```
AI-PhotoEditor-Web/
├── static/
│   ├── styles.css
│   └── app.js
├── templates/
│   └── index.html
├── uploads/              # uploaded images (sample included)
├── outputs/              # processed images + metadata (JSON)
├── app.py
├── image_processing.py
├── requirements.txt
└── README.md
```

## 🚀 Setup & Run

1. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run**
   ```bash
   python app.py
   ```

4. **Open your browser**
   - http://127.0.0.1:5000/

## 🖱️ How to Use
- **Basic** tab → choose group and method, set parameters, hit **Process** or just tweak and watch the live preview update.
- **Advanced** tab → pick an advanced method. For histograms, enable **Show histograms**; the before/after plots appear below the form.
- Click **Download result** to save the output image; corresponding **metadata JSON** is stored in `outputs/`.

## 🧠 Educational Notes
- **Negative**: `s = 255 - r` (enhances white-on-black details).
- **Log**: `s = c * log(1 + r)` emphasizes darker regions; useful for low-intensity details.
- **Gamma**: `s = c * r^γ`. γ < 1 brightens, γ > 1 darkens.
- **Piecewise-linear**: flexible contrast stretching using breakpoints `(r1, s1)` and `(r2, s2)`.
- **HE vs CLAHE**: HE equalizes globally; CLAHE limits local contrast to reduce noise amplification.

## 🧹 Cleanup
You can safely delete files inside `uploads/` and `outputs/` at any time.

## 🐛 Troubleshooting
- If OpenCV can't read a particular image, the app falls back to Pillow for decoding.
- Large images are supported (up to 25MB by default).

---

MIT License © 2025


## 🚀 Smart Enhancement Presets (NEW)
Click once to apply a tuned pipeline (multiple filters in sequence). The UI shows intermediate stages, and you can still download the final image.

### Presets & Pipelines
1. **Sharpen Blurry Image**
   - Bilateral (d=7, σColor=50, σSpace=50) → Unsharp (k=5, σ=1.2, amount=1.6) → *(optional)* Gamma(γ=0.85 if image is dark)

2. **Enhance Low-light Image**
   - Gamma(γ=0.6) → CLAHE(clip=2.5, tile=8) → Unsharp (k=3, σ=1.0, amount=0.8)

3. **Denoise & Smooth**
   - Median(k=5) → Gaussian(k=3, σ=0.8) → Global Histogram Equalization

4. **Boost Contrast**
   - Piecewise-linear contrast (60→20, 200→235) → Sobel edge overlay (α=0.2)

5. **Artistic/HDR Effect**
   - HistEq → Unsharp (k=5, σ=1.0, amount=1.2) → Gamma(γ=0.9) → Laplacian edge overlay (α=0.25)

6. **Medical Image Clarity**
   - CLAHE(clip=2.0, tile=8) → Laplacian edge overlay (α=0.15) → Unsharp (k=3, σ=1.0, amount=0.6)

### Auto Suggest (Optional)
When you upload an image, the app analyzes brightness, contrast, sharpness, and dynamic range and may **recommend a preset**.

### Where is the logic?
- `presets.py` contains the pipeline logic and the image analyzer.
- `/preset` endpoint applies the chosen pipeline.
- `/suggest` endpoint recommends a preset based on image metrics.

