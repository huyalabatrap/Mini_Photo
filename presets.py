import cv2
import numpy as np
from image_processing import (
    to_bgr, to_gray, ensure_odd,
    mean_blur, gaussian_blur, median_blur, bilateral_filter,
    unsharp_mask, laplacian_sharpen,
    sobel_edges, laplacian_edges, canny_edges,
    hist_equalize, clahe_equalize,
    gamma_correction, log_transform, piecewise_linear, negative_image
)

def analyze_image(bgr):
    gray = to_gray(bgr)
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    # Sharpness via variance of Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())
    # Contrast proxy via percentile spread
    p5, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p5)
    return {
        "mean": round(mean_intensity, 2),
        "std": round(std_intensity, 2),
        "sharpness": round(sharpness, 2),
        "dynamic_range": round(dynamic_range, 2),
    }

def suggest_preset(metrics):
    mean_v = metrics["mean"]
    std_v = metrics["std"]
    sharp = metrics["sharpness"]
    dyn = metrics["dynamic_range"]

    if mean_v < 90 and dyn < 80:
        return "enhance_low_light"
    if sharp < 100:  # empirically chosen
        return "sharpen_blurry"
    if std_v > 60 and dyn > 120:
        return "denoise_smooth"
    if dyn < 70:
        return "boost_contrast"
    return "sharpen_blurry"

def _overlay_edges(bgr, edges_gray, alpha=0.25):
    edges_bgr = to_bgr(edges_gray)
    out = cv2.addWeighted(bgr.astype(np.float32), 1.0, edges_bgr.astype(np.float32), alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def run_pipeline(bgr, preset_name: str):
    """
    Returns: final_bgr, stages (list of tuples (label, bgr_image)), filters_used (list of strings)
    """
    name = preset_name.lower()
    stages = []
    filters = []

    if name in ("sharpen_blurry", "sharpen"):
        # Denoise slightly to avoid sharpening noise
        s1 = bilateral_filter(bgr, diameter=7, sigma_color=50, sigma_space=50)
        stages.append(("Bilateral (d7, σC50, σS50)", s1)); filters.append("Bilateral(d=7,σColor=50,σSpace=50)")

        # Unsharp
        s2 = unsharp_mask(s1, kernel_size=5, sigma=1.2, amount=1.6)
        stages.append(("Unsharp (k=5, σ=1.2, amt=1.6)", s2)); filters.append("Unsharp(k=5,σ=1.2,amount=1.6)")

        # If image is dark, gently brighten
        metrics = analyze_image(s2)
        if metrics["mean"] < 100:
            s3 = gamma_correction(s2, gamma=0.85, c=1.0)
            stages.append(("Gamma Correction (γ=0.85)", s3)); filters.append("Gamma(γ=0.85)")
        else:
            s3 = s2

        final = s3

    elif name in ("enhance_low_light", "low_light"):
        s1 = gamma_correction(bgr, gamma=0.6, c=1.0)
        stages.append(("Gamma Correction (γ=0.6)", s1)); filters.append("Gamma(γ=0.6)")

        s2 = clahe_equalize(s1, clip_limit=2.5, tile_grid_size=8)
        stages.append(("CLAHE (clip=2.5, tile=8)", s2)); filters.append("CLAHE(clip=2.5,tile=8)")

        s3 = unsharp_mask(s2, kernel_size=3, sigma=1.0, amount=0.8)
        stages.append(("Unsharp (k=3, σ=1.0, amt=0.8)", s3)); filters.append("Unsharp(k=3,σ=1.0,amount=0.8)")

        final = s3

    elif name in ("denoise_smooth", "denoise"):
        s1 = median_blur(bgr, kernel_size=5)
        stages.append(("Median Blur (k=5)", s1)); filters.append("Median(k=5)")

        s2 = gaussian_blur(s1, kernel_size=3, sigma=0.8)
        stages.append(("Gaussian (k=3, σ=0.8)", s2)); filters.append("Gaussian(k=3,σ=0.8)")

        s3 = hist_equalize(s2)
        stages.append(("Histogram Equalization", s3)); filters.append("HistEqualize")

        final = s3

    elif name in ("boost_contrast", "contrast"):
        s1 = piecewise_linear(bgr, r1=60, s1=20, r2=200, s2=235)
        stages.append(("Piecewise Linear (60→20, 200→235)", s1)); filters.append("Piecewise(r1=60,s1=20,r2=200,s2=235)")

        s2_edges = sobel_edges(s1, kernel_size=3)
        s2 = _overlay_edges(s1, s2_edges, alpha=0.20)
        stages.append(("Sobel Overlay (α=0.2)", s2)); filters.append("SobelOverlay(α=0.2)")

        final = s2

    elif name in ("artistic_hdr", "hdr"):
        s1 = hist_equalize(bgr)
        stages.append(("Histogram Equalization", s1)); filters.append("HistEqualize")

        s2 = unsharp_mask(s1, kernel_size=5, sigma=1.0, amount=1.2)
        stages.append(("Unsharp (k=5, σ=1.0, amt=1.2)", s2)); filters.append("Unsharp(k=5,σ=1.0,amount=1.2)")

        s3 = gamma_correction(s2, gamma=0.9, c=1.0)
        stages.append(("Gamma Correction (γ=0.9)", s3)); filters.append("Gamma(γ=0.9)")

        edges = laplacian_edges(s3, kernel_size=3)
        s4 = _overlay_edges(s3, edges, alpha=0.25)
        stages.append(("Laplacian Overlay (α=0.25)", s4)); filters.append("LaplacianOverlay(α=0.25)")

        final = s4

    elif name in ("medical_clarity", "medical"):
        # Medical images are often grayscale; emphasize fine detail without overshooting
        s1 = clahe_equalize(bgr, clip_limit=2.0, tile_grid_size=8)
        stages.append(("CLAHE (clip=2.0, tile=8)", s1)); filters.append("CLAHE(clip=2.0,tile=8)")

        edges = laplacian_edges(s1, kernel_size=3)
        s2 = _overlay_edges(s1, edges, alpha=0.15)
        stages.append(("Laplacian Overlay (α=0.15)", s2)); filters.append("LaplacianOverlay(α=0.15)")

        s3 = unsharp_mask(s2, kernel_size=3, sigma=1.0, amount=0.6)
        stages.append(("Unsharp (k=3, σ=1.0, amt=0.6)", s3)); filters.append("Unsharp(k=3,σ=1.0,amount=0.6)")

        final = s3

    else:
        # Fallback: identity
        final = bgr
        stages.append(("Original", bgr.copy()))
        filters.append("None")

    return final, stages, filters
