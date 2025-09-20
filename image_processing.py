import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_odd(k: int) -> int:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return k

def to_gray(bgr):
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# ---------- Intensity Transformations ----------
def negative_image(bgr):
    return 255 - bgr

def log_transform(bgr, c=1.0):
    """
    s = c * log(1 + r) scaled to [0,255]. Operates on grayscale then converted to BGR.
    """
    gray = to_gray(bgr).astype(np.float32)
    # Normalize to [0,1], apply log, then scale back
    gray_norm = gray / 255.0
    s = c * np.log1p(gray_norm)
    s = s / s.max() * 255.0
    return to_bgr(s.astype(np.uint8))

def gamma_correction(bgr, gamma=1.0, c=1.0):
    """
    s = c * r^gamma with r in [0,1], scaled back to [0,255]. Grayscale then BGR.
    """
    gray = to_gray(bgr).astype(np.float32) / 255.0
    s = c * np.power(gray, max(gamma, 1e-6))
    s = np.clip(s / (s.max() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return to_bgr(s)

def piecewise_linear(bgr, r1=50, s1=0, r2=200, s2=255):
    """
    Three-segment piecewise linear transform on grayscale:
    (0,0) -> (r1,s1); (r1,s1) -> (r2,s2); (r2,s2) -> (255,255)
    """
    r1 = np.clip(int(r1), 0, 255)
    r2 = np.clip(int(r2), 0, 255)
    s1 = np.clip(int(s1), 0, 255)
    s2 = np.clip(int(s2), 0, 255)
    if r2 <= r1: r2 = min(255, r1 + 1)

    gray = to_gray(bgr).astype(np.float32)

    def segment(x, x0, y0, x1, y1):
        # Linear mapping on [x0,x1]
        m = (y1 - y0) / max(1e-6, (x1 - x0))
        return m * (x - x0) + y0

    out = np.zeros_like(gray, dtype=np.float32)
    # Segment 1
    mask1 = gray <= r1
    out[mask1] = segment(gray[mask1], 0, 0, r1, s1)
    # Segment 2
    mask2 = (gray > r1) & (gray <= r2)
    out[mask2] = segment(gray[mask2], r1, s1, r2, s2)
    # Segment 3
    mask3 = gray > r2
    out[mask3] = segment(gray[mask3], r2, s2, 255, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return to_bgr(out)

# ---------- Histogram Equalization ----------
def hist_equalize(bgr):
    """Global histogram equalization on grayscale."""
    gray = to_gray(bgr)
    eq = cv2.equalizeHist(gray)
    return to_bgr(eq)

def clahe_equalize(bgr, clip_limit=2.0, tile_grid_size=8):
    """CLAHE on grayscale."""
    gray = to_gray(bgr)
    tile_grid_size = max(1, int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid_size, tile_grid_size))
    eq = clahe.apply(gray)
    return to_bgr(eq)

# ---------- Blur Filters (enhanced) ----------
def mean_blur(bgr, kernel_size=3):
    k = ensure_odd(kernel_size)
    return cv2.blur(bgr, (k, k))

def gaussian_blur(bgr, kernel_size=3, sigma=1.0):
    k = ensure_odd(kernel_size)
    return cv2.GaussianBlur(bgr, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))

def median_blur(bgr, kernel_size=3):
    k = ensure_odd(kernel_size)
    if k < 3: k = 3
    return cv2.medianBlur(bgr, k)

def bilateral_filter(bgr, diameter=9, sigma_color=75, sigma_space=75):
    d = int(diameter)
    return cv2.bilateralFilter(bgr, d, float(sigma_color), float(sigma_space))

# ---------- Edge Detection ----------
def sobel_edges(bgr, kernel_size=3):
    kernel_size = int(kernel_size)
    if kernel_size not in (1, 3, 5, 7):
        kernel_size = 3
    gray = to_gray(bgr)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = np.clip((mag / (mag.max() + 1e-6)) * 255, 0, 255).astype(np.uint8)
    return mag

def prewitt_edges(bgr):
    gray = to_gray(bgr).astype(np.float32)
    # 3x3 Prewitt kernels
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[ 1,  1,  1],
                   [ 0,  0,  0],
                   [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip((mag / (mag.max() + 1e-6)) * 255, 0, 255).astype(np.uint8)
    return mag

def laplacian_edges(bgr, kernel_size=3):
    kernel_size = ensure_odd(kernel_size)
    gray = to_gray(bgr)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=kernel_size)
    lap_abs = cv2.convertScaleAbs(lap)
    return lap_abs

def canny_edges(bgr, threshold1=100, threshold2=200):
    gray = to_gray(bgr)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

def binary_threshold(bgr, threshold=128):
    gray = to_gray(bgr)
    _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return binary

# ---------- Sharpening ----------
def unsharp_mask(bgr, kernel_size=3, sigma=1.0, amount=1.5):
    kernel_size = ensure_odd(kernel_size)
    blurred = cv2.GaussianBlur(bgr, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr.astype(np.float32), 1 + amount, blurred.astype(np.float32), -amount, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

def laplacian_sharpen(bgr, kernel_size=3, alpha=1.0):
    kernel_size = ensure_odd(kernel_size)
    gray = to_gray(bgr)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=kernel_size)
    lap_abs = cv2.convertScaleAbs(lap)
    lap_bgr = to_bgr(lap_abs)
    sharp = cv2.addWeighted(bgr.astype(np.float32), 1.0, lap_bgr.astype(np.float32), -alpha, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp

# ---------- Dispatcher ----------
def apply_filter(image_bgr, group='blur', method='gaussian', **kwargs):
    """
    image_bgr: np.ndarray (H, W, 3) BGR uint8
    group: 'blur' | 'sharpen' | 'edge' | 'advanced'
    """
    ksize = ensure_odd(int(kwargs.get('kernel_size', 3)))
    sigma = float(kwargs.get('sigma', 1.0))
    threshold = float(kwargs.get('threshold', 128))
    threshold1 = float(kwargs.get('threshold1', 100))
    threshold2 = float(kwargs.get('threshold2', 200))
    amount = float(kwargs.get('amount', 1.5))
    alpha  = float(kwargs.get('alpha', 1.0))

    # Advanced params
    gamma = float(kwargs.get('gamma', 1.0))
    c = float(kwargs.get('c', 1.0))
    r1 = float(kwargs.get('r1', 50))
    s1 = float(kwargs.get('s1', 0))
    r2 = float(kwargs.get('r2', 200))
    s2 = float(kwargs.get('s2', 255))
    clip_limit = float(kwargs.get('clip_limit', 2.0))
    tile_grid_size = int(kwargs.get('tile_grid_size', 8))
    diameter = int(kwargs.get('diameter', 9))
    sigma_color = float(kwargs.get('sigma_color', 75))
    sigma_space = float(kwargs.get('sigma_space', 75))

    if group == 'blur':
        m = method.lower()
        if m == 'mean':
            out = mean_blur(image_bgr, ksize)
        elif m == 'gaussian':
            out = gaussian_blur(image_bgr, ksize, sigma)
        elif m == 'median':
            out = median_blur(image_bgr, ksize)
        elif m == 'bilateral':
            out = bilateral_filter(image_bgr, diameter, sigma_color, sigma_space)
        else:
            out = image_bgr

    elif group == 'sharpen':
        m = method.lower()
        if m == 'unsharp':
            out = unsharp_mask(image_bgr, kernel_size=ksize, sigma=sigma, amount=amount)
        elif m == 'laplacian_sharpen':
            out = laplacian_sharpen(image_bgr, kernel_size=ksize, alpha=alpha)
        else:
            out = image_bgr

    elif group == 'edge':
        m = method.lower()
        if m == 'sobel':
            out = to_bgr(sobel_edges(image_bgr, kernel_size=ksize))
        elif m == 'prewitt':
            out = to_bgr(prewitt_edges(image_bgr))
        elif m == 'laplacian':
            out = to_bgr(laplacian_edges(image_bgr, kernel_size=ksize))
        elif m == 'canny':
            out = to_bgr(canny_edges(image_bgr, threshold1=threshold1, threshold2=threshold2))
        elif m == 'binary':
            out = to_bgr(binary_threshold(image_bgr, threshold=threshold))
        else:
            out = image_bgr

    elif group == 'advanced':
        m = method.lower()
        if m == 'negative':
            out = negative_image(image_bgr)
        elif m == 'log':
            out = log_transform(image_bgr, c=c)
        elif m == 'gamma':
            out = gamma_correction(image_bgr, gamma=gamma, c=c)
        elif m == 'piecewise':
            out = piecewise_linear(image_bgr, r1=r1, s1=s1, r2=r2, s2=s2)
        elif m == 'hist_eq':
            out = hist_equalize(image_bgr)
        elif m == 'clahe':
            out = clahe_equalize(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        else:
            out = image_bgr

    else:
        out = image_bgr

    return out
