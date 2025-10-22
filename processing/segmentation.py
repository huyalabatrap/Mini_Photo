"""
Image segmentation processing module.
Implements various segmentation methods for real-world applications.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Tuple


@dataclass
class SegmentationResult:
    """Data class for segmentation results."""
    job_id: str
    method: str
    result_path: Path
    overlay_path: Optional[Path]
    cutout_path: Optional[Path]
    stats: Dict


def run_segmentation(
    image_path: Union[str, Path], 
    method: str, 
    params: Optional[Dict] = None, 
    job_id: Optional[str] = None
) -> SegmentationResult:
    """
    Run image segmentation using specified method.
    
    Args:
        image_path: Path to input image
        method: Segmentation method name
        params: Additional parameters for the method
        job_id: Unique job identifier
        
    Returns:
        SegmentationResult containing paths and statistics
        
    Raises:
        ValueError: If image cannot be read or method is invalid
    """
    if params is None:
        params = {}
    
    if job_id is None:
        job_id = "default"
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")
    
    # Select segmentation method
    method_map = {
        'background': _segment_background,
        'count': _segment_count_objects,
        'skin': _segment_skin,
        'plate': _segment_license_plate,
        'lesion': _segment_lesion,
        'otsu': _segment_otsu,
        'kmeans': _segment_kmeans,
        'canny': _segment_canny,
        'watershed': _segment_watershed
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Run segmentation
    mask, stats = method_map[method](image, params)
    
    # Ensure mask is 8-bit binary (0/255)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Create output paths
    results_dir = Path("static/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = results_dir / f"{job_id}_mask.png"
    overlay_path = results_dir / f"{job_id}_overlay.png"
    cutout_path = results_dir / f"{job_id}_cutout.png"
    
    # Save mask
    cv2.imwrite(str(mask_path), mask)
    
    # Create overlay (red contours on original image)
    overlay = _create_overlay(image, mask)
    cv2.imwrite(str(overlay_path), overlay)
    
    # Create cutout with transparent background
    cutout = _create_cutout(image, mask)
    cv2.imwrite(str(cutout_path), cutout)
    
    # Calculate final statistics
    final_stats = _calculate_stats(mask, stats)
    
    return SegmentationResult(
        job_id=job_id,
        method=method,
        result_path=mask_path,
        overlay_path=overlay_path,
        cutout_path=cutout_path,
        stats=final_stats
    )


def _segment_background(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Remove background using GrabCut algorithm with improved initialization."""
    height, width = image.shape[:2]
    
    # For license plate images, use a smaller, more centered rectangle
    margin_x = 0.2  # 20% margin on sides
    margin_y = 0.25  # 25% margin on top/bottom
    
    rect = (
        int(width * margin_x), 
        int(height * margin_y),
        int(width * (1 - 2 * margin_x)), 
        int(height * (1 - 2 * margin_y))
    )
    
    # Initialize masks for GrabCut
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        # Apply GrabCut with more iterations for better results
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
        
        # Extract foreground (probable foreground + definite foreground)
        mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Post-processing to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill holes
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # If mask is mostly empty, try different approach
        if np.sum(mask_final) < (height * width * 0.05):  # Less than 5% of image
            # Fallback: use thresholding on center region
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            center_region = gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            
            # Apply adaptive threshold
            _, thresh_center = cv2.threshold(center_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Create mask for center region
            mask_final = np.zeros((height, width), np.uint8)
            mask_final[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = thresh_center
            
            # Clean up
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=1)
        
    except cv2.error:
        # Fallback if GrabCut fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask_final = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_final = (mask_final > 0).astype(np.uint8)
    
    return mask_final * 255, {"method_specific": "grabcut_background_removal_improved"}


def _segment_count_objects(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced object counting using watershed segmentation.
    Detects and counts separate objects (coins, cells, etc.)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold for better object detection
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Remove small noise
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_noise, iterations=2)
    
    # Distance transform to find centers
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Find local maxima as seeds
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Find background region
    sure_bg = cv2.dilate(opening, kernel_noise, iterations=3)
    
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so background is not 0, but 1
    markers = markers + 1
    
    # Mark unknown regions as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    image_for_watershed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(image_for_watershed, markers)
    
    # Create mask from watershed result
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255  # Exclude background (1) and boundaries (-1)
    
    # Count objects
    num_objects = len(np.unique(markers)) - 2  # Exclude background and boundaries
    num_objects = max(0, num_objects)
    
    # Additional statistics
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours
    min_area = 50
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    num_objects = len(valid_contours)
    
    # Calculate average object size
    if valid_contours:
        areas = [cv2.contourArea(c) for c in valid_contours]
        avg_area = np.mean(areas)
        max_area = np.max(areas)
        min_area_stat = np.min(areas)
    else:
        avg_area = 0
        max_area = 0
        min_area_stat = 0
    
    return mask, {
        "num_regions": num_objects,
        "method_specific": "watershed_counting_advanced",
        "avg_object_area": round(avg_area, 2),
        "largest_object": int(max_area),
        "smallest_object": int(min_area_stat)
    }


def _segment_skin(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced skin detection using multiple color spaces and strict filtering.
    Optimized to exclude hair, clothing, and background while detecting human skin.
    """
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to multiple color spaces
    ycrcb = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    bgr = filtered
    
    # Extract channels
    y, cr, cb = cv2.split(ycrcb)
    h, s, v = cv2.split(hsv)
    b_chan, g_chan, r_chan = cv2.split(bgr)
    
    # STRICT SKIN DETECTION RULES
    # Rule 1: YCrCb range (most important for skin)
    # Tighter ranges to exclude non-skin
    lower_ycrcb = np.array([20, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 170, 120], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Rule 2: HSV range - exclude very dark (hair) and very saturated colors (clothing)
    # Skin has moderate saturation and brightness
    lower_hsv = np.array([0, 20, 60], dtype=np.uint8)
    upper_hsv = np.array([20, 150, 245], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Rule 3: RGB conditions to exclude common non-skin colors
    # Exclude dark pixels (hair, dark clothing)
    mask_not_dark = cv2.threshold(v, 40, 255, cv2.THRESH_BINARY)[1]
    
    # Exclude very saturated colors (bright clothing)
    mask_not_saturated = cv2.threshold(s, 160, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Exclude blue/green dominant areas (clothing, background)
    # Skin typically has R > G > B
    r_greater_g = cv2.compare(r_chan, g_chan, cv2.CMP_GT)
    g_greater_b = cv2.compare(g_chan, b_chan, cv2.CMP_GT)
    mask_rgb_order = cv2.bitwise_and(r_greater_g, g_greater_b)
    
    # Rule 4: Cr-Cb relationship (critical for skin)
    # For skin: Cr - Cb > 10 (removes blue/green tints)
    cr_cb_diff = cv2.subtract(cr, cb)
    mask_cr_cb = cv2.threshold(cr_cb_diff, 15, 255, cv2.THRESH_BINARY)[1]
    
    # Rule 5: Specific Cr and Cb ranges
    mask_cr = cv2.inRange(cr, 140, 165)
    mask_cb = cv2.inRange(cb, 85, 115)
    
    # Rule 6: Exclude extreme brightness (glare, white clothing)
    mask_not_too_bright = cv2.threshold(v, 250, 255, cv2.THRESH_BINARY_INV)[1]
    
    # COMBINE ALL RULES WITH AND OPERATIONS (strict filtering)
    # Start with YCrCb (base)
    mask = mask_ycrcb.copy()
    
    # Apply HSV constraints
    mask = cv2.bitwise_and(mask, mask_hsv)
    
    # Apply darkness filter
    mask = cv2.bitwise_and(mask, mask_not_dark)
    
    # Apply saturation filter
    mask = cv2.bitwise_and(mask, mask_not_saturated)
    
    # Apply RGB order check
    mask = cv2.bitwise_and(mask, mask_rgb_order)
    
    # Apply Cr-Cb difference
    mask = cv2.bitwise_and(mask, mask_cr_cb)
    
    # Apply specific Cr range
    mask = cv2.bitwise_and(mask, mask_cr)
    
    # Apply specific Cb range
    mask = cv2.bitwise_and(mask, mask_cb)
    
    # Apply brightness filter
    mask = cv2.bitwise_and(mask, mask_not_too_bright)
    
    # Morphological operations - CONSERVATIVE to avoid including non-skin
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise with opening (aggressive)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=3)
    
    # Fill SMALL holes only (don't connect distant regions)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    # Apply median filter to smooth edges
    mask = cv2.medianBlur(mask, 5)
    
    # Erode slightly to remove edge pixels (often mixed with hair/clothing)
    mask = cv2.erode(mask, kernel_small, iterations=1)
    
    # Connected components analysis with STRICT filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create filtered mask
    mask_filtered = np.zeros_like(mask)
    valid_regions = []
    
    # Minimum area: 1% of image (remove small noise)
    min_area = (mask.shape[0] * mask.shape[1]) * 0.01
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Get region bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Filter by area
        if area < min_area:
            continue
        
        # Filter by aspect ratio (skin regions shouldn't be too elongated)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        if aspect_ratio > 5:  # Too elongated (likely artifact)
            continue
        
        # Filter by compactness (skin is relatively round/oval)
        # Compactness = area / (perimeter^2)
        region_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                compactness = (4 * np.pi * area) / (perimeter * perimeter)
                # Skin regions should have reasonable compactness (not too irregular)
                if compactness < 0.05:  # Too irregular
                    continue
        
        # Filter by density (% of pixels in bounding box)
        bbox_area = w * h
        density = area / (bbox_area + 1e-6)
        if density < 0.3:  # Too sparse (likely noise)
            continue
        
        # Passed all filters - add to mask
        mask_filtered[labels == i] = 255
        valid_regions.append(area)
    
    # Use filtered mask
    if len(valid_regions) > 0:
        mask = mask_filtered
        num_regions = len(valid_regions)
        avg_region_size = np.mean(valid_regions)
        largest_region = np.max(valid_regions)
    else:
        # No regions passed filters - return empty mask
        mask = np.zeros_like(mask)
        num_regions = 0
        avg_region_size = 0
        largest_region = 0
    
    # Final threshold to ensure binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask, {
        "method_specific": "strict_multi_rule_skin_detection",
        "num_skin_regions": num_regions,
        "avg_region_size": int(avg_region_size),
        "largest_region": int(largest_region),
        "detection_rules": "9 strict filters (YCrCb + HSV + RGB + Cr-Cb + shape)",
        "filters_applied": "darkness, saturation, RGB order, Cr-Cb diff, brightness, aspect ratio, compactness, density"
    }


def _segment_license_plate(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced license plate detection using multiple strategies.
    Detects rectangular regions with high contrast text patterns.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Create initial mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    all_candidates = []
    
    # STRATEGY 1: Edge-based detection with strong rectangular constraint
    # Find edges using Canny
    edges = cv2.Canny(filtered, 30, 200)
    
    # Dilate edges to connect nearby components
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_rect, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # License plate aspect ratio: typically 2:1 to 5:1 (width:height)
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        
        # Filter by size and aspect ratio
        if (w > 50 and h > 15 and 
            2.0 <= aspect_ratio <= 6.0 and 
            w < width * 0.8 and h < height * 0.5):
            
            area = w * h
            
            # Check rectangularity (how well it fits bounding box)
            contour_area = cv2.contourArea(contour)
            if contour_area > 0:
                rectangularity = contour_area / area
                if rectangularity > 0.5:  # Should be fairly rectangular
                    all_candidates.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'score': area * rectangularity,
                        'method': 'edge_detection'
                    })
    
    # STRATEGY 2: Morphological text detection
    # Top-hat transform to enhance bright text on dark background
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel_tophat)
    
    # Black-hat transform to enhance dark text on bright background
    blackhat = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_tophat)
    
    # Combine both
    morph_combined = cv2.add(tophat, blackhat)
    
    # Threshold
    _, morph_thresh = cv2.threshold(morph_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Close gaps between characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph_closed = cv2.morphologyEx(morph_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Find contours
    morph_contours, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in morph_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        
        if (w > 60 and h > 15 and 
            2.0 <= aspect_ratio <= 6.0):
            
            area = w * h
            all_candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': area,
                'method': 'morphological_text'
            })
    
    # STRATEGY 3: Sobel gradient + variance (text has high local variance)
    # Sobel in X direction (vertical edges - important for text)
    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(sobelx / sobelx.max() * 255)
    
    # Threshold sobel
    _, sobel_thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Close to connect text
    kernel_sobel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    sobel_closed = cv2.morphologyEx(sobel_thresh, cv2.MORPH_CLOSE, kernel_sobel)
    
    sobel_contours, _ = cv2.findContours(sobel_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in sobel_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        
        if (w > 70 and h > 18 and 
            2.0 <= aspect_ratio <= 6.0):
            
            # Calculate variance in region (text should have high variance)
            roi = filtered[y:y+h, x:x+w]
            variance = np.var(roi)
            
            if variance > 200:  # High contrast region
                area = w * h
                all_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'score': area * (variance / 1000),
                    'method': 'sobel_variance'
                })
    
    # STRATEGY 4: Color-based (white/yellow plates)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # White plates
    lower_white = np.array([0, 0, 170], dtype=np.uint8)
    upper_white = np.array([180, 30, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Yellow plates
    lower_yellow = np.array([15, 40, 140], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine color masks
    mask_color = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Morphology
    kernel_color = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_color, iterations=2)
    
    color_contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in color_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if h == 0:
            continue
        aspect_ratio = float(w) / h
        
        if (w > 80 and h > 20 and 
            2.0 <= aspect_ratio <= 6.0):
            
            area = w * h
            all_candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': area * 1.2,  # Boost color-based detection
                'method': 'color_detection'
            })
    
    # STRATEGY 5: Contour approximation (rectangles)
    # Find all contours in the image
    _, binary = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    all_contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in all_contours:
        # Approximate contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # License plates are typically 4-sided
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            
            if h == 0:
                continue
            aspect_ratio = float(w) / h
            
            if (w > 60 and h > 15 and 
                2.0 <= aspect_ratio <= 6.0):
                
                area = w * h
                all_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'score': area,
                    'method': 'rectangle_approximation'
                })
    
    # EVALUATE AND SELECT BEST CANDIDATE
    if not all_candidates:
        # No candidates found - return empty mask
        return mask, {
            "method_specific": "license_plate_no_detection",
            "detected_area": 0,
            "aspect_ratio": 0,
            "plate_dimensions": "0x0",
            "strategies_tried": 5
        }
    
    # Score candidates based on multiple criteria
    for candidate in all_candidates:
        x, y, w, h = candidate['bbox']
        
        # Extract ROI
        roi = filtered[y:y+h, x:x+w]
        
        # Calculate additional features
        # 1. Edge density (plates have high edge density from text)
        roi_edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(roi_edges > 0) / (w * h)
        
        # 2. Variance (text creates high variance)
        variance = np.var(roi) / 1000
        
        # 3. Aspect ratio score (closer to 3.0 is better)
        ideal_ratio = 3.5
        aspect_score = 1.0 / (1.0 + abs(candidate['aspect_ratio'] - ideal_ratio))
        
        # 4. Position score (plates often in lower half of image)
        position_score = 1.0 + (y / height) * 0.5
        
        # Combined score
        candidate['final_score'] = (
            candidate['score'] * 0.3 +
            edge_density * 1000 * 0.25 +
            variance * 100 * 0.15 +
            aspect_score * 500 * 0.15 +
            position_score * 200 * 0.15
        )
    
    # Sort by final score
    all_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Get best candidate
    best = all_candidates[0]
    x, y, w, h = best['bbox']
    
    # Create mask with expanded region (add small margin)
    margin = 5
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)
    
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    return mask, {
        "method_specific": f"license_plate_{best['method']}",
        "detected_area": int(best['area']),
        "aspect_ratio": round(best['aspect_ratio'], 2),
        "plate_dimensions": f"{w}x{h}",
        "confidence_score": round(best['final_score'], 2),
        "candidates_found": len(all_candidates),
        "strategies_used": "5 (edges, morphology, sobel, color, rectangles)"
    }


def _segment_lesion(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced lesion/wound detection using color and texture analysis.
    Detects reddish wounds, bruises, and skin abnormalities.
    """
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Convert to multiple color spaces
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(filtered, cv2.COLOR_BGR2YCrCb)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    l, a, b_lab = cv2.split(lab)
    y, cr, cb = cv2.split(ycrcb)
    b_chan, g_chan, r_chan = cv2.split(filtered)
    
    # STRATEGY 1: Red/Pink wound detection (fresh wounds, inflammation)
    # HSV ranges for red/pink wounds
    # Red wraps around in HSV (0-10 and 170-180)
    lower_red1 = np.array([0, 30, 30], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 30, 30], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # STRATEGY 2: Dark/brown lesion detection (scabs, bruises)
    # Detect darker regions with reddish-brown tones
    lower_dark = np.array([0, 20, 20], dtype=np.uint8)
    upper_dark = np.array([20, 150, 120], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # STRATEGY 3: LAB color space (A channel is good for red detection)
    # High 'a' value indicates red
    mask_lab_a = cv2.threshold(a, 140, 255, cv2.THRESH_BINARY)[1]
    
    # STRATEGY 4: YCrCb - Cr channel highlights red
    mask_cr = cv2.threshold(cr, 160, 255, cv2.THRESH_BINARY)[1]
    
    # STRATEGY 5: RGB dominance - wounds often have R > G significantly
    r_minus_g = cv2.subtract(r_chan, g_chan)
    mask_r_dominance = cv2.threshold(r_minus_g, 15, 255, cv2.THRESH_BINARY)[1]
    
    # STRATEGY 6: Texture-based detection using gradient
    # Wounds often have distinct edges/texture
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    gradient = np.uint8(gradient / gradient.max() * 255)
    mask_gradient = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)[1]
    
    # STRATEGY 7: Statistical anomaly detection
    # Find regions that deviate from mean skin tone
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Detect darker or lighter anomalies
    mask_darker = cv2.threshold(gray, int(mean_val - std_val * 0.8), 255, cv2.THRESH_BINARY_INV)[1]
    mask_lighter = cv2.threshold(gray, int(mean_val + std_val * 0.8), 255, cv2.THRESH_BINARY)[1]
    mask_anomaly = cv2.bitwise_or(mask_darker, mask_lighter)
    
    # COMBINE STRATEGIES with weighted approach
    # Red detection is most important for wounds
    mask_combined = cv2.addWeighted(mask_red, 0.35, mask_dark, 0.15, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_lab_a, 0.15, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_cr, 0.15, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_r_dominance, 0.10, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_gradient, 0.05, 0)
    mask_combined = cv2.addWeighted(mask_combined, 1.0, mask_anomaly, 0.05, 0)
    
    # Threshold to binary
    _, mask = cv2.threshold(mask_combined, 80, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Fill holes in lesions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    
    # Dilate slightly to ensure coverage
    mask = cv2.dilate(mask, kernel_medium, iterations=1)
    
    # Find contours and filter
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros_like(gray), {
            "method_specific": "multi_strategy_lesion_detection",
            "lesions_detected": 0,
            "detection_methods": "7 strategies (red, dark, LAB, YCrCb, RGB, gradient, anomaly)"
        }
    
    # Filter contours by size and shape
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * 0.0005  # 0.05% minimum
    max_area = image_area * 0.5     # 50% maximum
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by circularity (lesions are somewhat round)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        
        # Accept if reasonably circular (0.1 to 1.0)
        if circularity > 0.05:  # Very lenient for irregular wounds
            valid_contours.append((contour, area, circularity))
    
    # Sort by area (largest first)
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    
    # Create final mask with valid lesions
    mask_final = np.zeros_like(gray)
    
    if valid_contours:
        # Draw all valid contours
        for contour, _, _ in valid_contours:
            cv2.drawContours(mask_final, [contour], -1, 255, -1)
        
        num_lesions = len(valid_contours)
        total_area = sum(area for _, area, _ in valid_contours)
        largest_area = valid_contours[0][1]
        avg_circularity = np.mean([circ for _, _, circ in valid_contours])
    else:
        num_lesions = 0
        total_area = 0
        largest_area = 0
        avg_circularity = 0
    
    return mask_final, {
        "method_specific": "multi_strategy_lesion_detection",
        "lesions_detected": num_lesions,
        "total_lesion_area": int(total_area),
        "largest_lesion": int(largest_area),
        "avg_circularity": round(avg_circularity, 3),
        "detection_methods": "Red/pink detection + Dark lesions + LAB + YCrCb + RGB + Gradient + Anomaly"
    }


def _segment_otsu(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Improved Otsu thresholding with preprocessing for text detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different approaches and choose the best one
    
    # Approach 1: Standard Otsu
    _, mask1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Approach 2: Inverted Otsu (for white text on dark background)
    _, mask2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Approach 3: Adaptive threshold
    mask3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Choose the mask with more reasonable amount of foreground
    masks = [mask1, mask2, mask3]
    mask_scores = []
    
    for mask in masks:
        # Calculate foreground ratio
        fg_ratio = np.sum(mask > 0) / mask.size
        
        # Prefer masks with 10-70% foreground (reasonable for most images)
        if 0.1 <= fg_ratio <= 0.7:
            score = 1.0 - abs(0.3 - fg_ratio)  # Prefer ~30% foreground
        else:
            score = 0.0
        
        mask_scores.append(score)
    
    # Select best mask
    best_idx = np.argmax(mask_scores)
    mask = masks[best_idx]
    
    # If no good mask found, use standard Otsu
    if mask_scores[best_idx] == 0:
        mask = mask1
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # For text images, also try to connect nearby characters
    if best_idx in [0, 1]:  # If using Otsu variants
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, connect_kernel, iterations=1)
    
    method_type = ["otsu_standard", "otsu_inverted", "adaptive_threshold"][best_idx]
    
    return mask, {
        "method_specific": f"otsu_thresholding_{method_type}",
        "threshold_type": method_type,
        "foreground_ratio": round(np.sum(mask > 0) / mask.size, 3)
    }


def _segment_kmeans(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced K-means color clustering with intelligent foreground selection.
    Segments image based on color similarity.
    """
    # Get number of clusters
    k = params.get('kmeans_k', 3)
    k = max(2, min(6, k))  # Clamp between 2-6
    
    # Preprocess: Apply bilateral filter to reduce noise while preserving edges
    preprocessed = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Reshape image for clustering
    pixel_values = preprocessed.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    # Create segmented image
    segmented_image = centers[labels].reshape(image.shape)
    
    # Analyze each cluster to find foreground
    cluster_scores = []
    
    for cluster_id in range(k):
        cluster_mask = (labels == cluster_id).reshape(image.shape[:2])
        
        # Calculate cluster properties
        cluster_pixels = preprocessed[cluster_mask]
        
        if len(cluster_pixels) > 0:
            # Convert to HSV for better color analysis
            cluster_color = centers[cluster_id]
            cluster_hsv = cv2.cvtColor(np.uint8([[cluster_color]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Scoring factors:
            # 1. Saturation (more saturated = more likely foreground)
            saturation_score = cluster_hsv[1] / 255.0
            
            # 2. Value/Brightness (moderate brightness preferred)
            value = cluster_hsv[2] / 255.0
            value_score = 1.0 - abs(value - 0.5) * 2  # Prefer mid-range brightness
            
            # 3. Cluster size (not too small, not too large)
            size_ratio = np.sum(cluster_mask) / cluster_mask.size
            if 0.1 <= size_ratio <= 0.7:
                size_score = 1.0
            elif size_ratio < 0.1:
                size_score = 0.3
            else:
                size_score = 0.5
            
            # 4. Spatial distribution (more centered = more likely foreground)
            y_coords, x_coords = np.where(cluster_mask)
            if len(y_coords) > 0:
                center_y, center_x = np.mean(y_coords), np.mean(x_coords)
                img_center_y, img_center_x = image.shape[0] / 2, image.shape[1] / 2
                
                # Distance from image center (normalized)
                dist = np.sqrt((center_y - img_center_y)**2 + (center_x - img_center_x)**2)
                max_dist = np.sqrt(img_center_y**2 + img_center_x**2)
                spatial_score = 1.0 - (dist / max_dist)
            else:
                spatial_score = 0.0
            
            # Combined score with weights
            total_score = (
                saturation_score * 0.35 +
                value_score * 0.25 +
                size_score * 0.25 +
                spatial_score * 0.15
            )
            
            cluster_scores.append((cluster_id, total_score, size_ratio))
    
    # Sort by score
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select foreground cluster(s)
    # Take the top 1-2 clusters with good scores
    foreground_clusters = []
    for cluster_id, score, size_ratio in cluster_scores:
        if score > 0.4 and len(foreground_clusters) < 2:  # Threshold for foreground
            foreground_clusters.append(cluster_id)
    
    # If no good cluster found, take the best one
    if not foreground_clusters and cluster_scores:
        foreground_clusters = [cluster_scores[0][0]]
    
    # Create final mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cluster_id in foreground_clusters:
        cluster_mask = (labels == cluster_id).reshape(image.shape[:2])
        mask[cluster_mask] = 255
    
    # Post-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask, {
        "method_specific": f"kmeans_intelligent_k{k}",
        "clusters": k,
        "foreground_clusters": len(foreground_clusters),
        "selected_clusters": foreground_clusters
    }


def _segment_canny(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced edge-based segmentation using Canny with adaptive thresholds.
    Creates regions from detected edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get Canny parameters or use adaptive values
    low_threshold = params.get('canny_low', None)
    high_threshold = params.get('canny_high', None)
    
    # If thresholds not provided, calculate automatically
    if low_threshold is None or high_threshold is None:
        # Use median-based automatic threshold
        v = np.median(gray)
        sigma = 0.33
        low_threshold = int(max(0, (1.0 - sigma) * v))
        high_threshold = int(min(255, (1.0 + sigma) * v))
        
        # Ensure reasonable range
        low_threshold = max(20, min(100, low_threshold))
        high_threshold = max(50, min(200, high_threshold))
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply Canny edge detection
    edges = cv2.Canny(filtered, low_threshold, high_threshold)
    
    # Edge enhancement - dilate first to make edges thicker
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)
    
    # Morphological closing to connect nearby edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    
    # Fill enclosed regions
    # Find contours
    contours, hierarchy = cv2.findContours(edges_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask by filling contours
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if contours:
        # Fill all contours
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Only fill reasonably sized regions
            if area > 100:  # Minimum area threshold
                cv2.drawContours(mask, contours, i, 255, -1)
    
    # If mask is mostly empty, use the closed edges directly
    if np.sum(mask > 0) < (mask.size * 0.05):  # Less than 5%
        mask = edges_closed
    
    # Final cleanup
    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final, iterations=2)
    
    # Remove very small regions
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    return mask, {
        "method_specific": f"canny_adaptive_low{low_threshold}_high{high_threshold}",
        "low_threshold": int(low_threshold),
        "high_threshold": int(high_threshold),
        "threshold_method": "automatic" if params.get('canny_low') is None else "manual"
    }


def _segment_watershed(image: np.ndarray, params: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Advanced watershed segmentation with marker-based initialization.
    Separates touching objects effectively.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing with bilateral filter
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Sure background area
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(opening, kernel_bg, iterations=3)
    
    # Distance transform to find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so that background is not 0, but 1
    markers = markers + 1
    
    # Mark the unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed on color image
    image_for_watershed = image.copy()
    markers = cv2.watershed(image_for_watershed, markers)
    
    # Create segmentation mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    # Mark all regions except background (1) and boundaries (-1)
    mask[markers > 1] = 255
    
    # Count regions
    unique_markers = np.unique(markers)
    num_regions = len(unique_markers[unique_markers > 1])  # Exclude background and boundaries
    
    # Additional cleanup
    kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_cleanup, iterations=1)
    
    # Calculate region statistics
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        avg_region_size = np.mean(areas)
        total_region_area = sum(areas)
    else:
        avg_region_size = 0
        total_region_area = 0
    
    return mask, {
        "method_specific": "watershed_marker_based",
        "num_regions": num_regions,
        "avg_region_size": round(avg_region_size, 2),
        "total_segmented_area": int(total_region_area)
    }


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create overlay image with red contours on original image."""
    overlay = image.copy()
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw red contours
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    
    return overlay


def _create_cutout(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create RGBA cutout with transparent background."""
    # Create 4-channel image (BGRA)
    cutout = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    
    # Copy BGR channels
    cutout[:, :, :3] = image
    
    # Set alpha channel based on mask
    cutout[:, :, 3] = mask
    
    return cutout


def _calculate_stats(mask: np.ndarray, method_stats: Dict) -> Dict:
    """Calculate final statistics for the segmentation result."""
    # Count foreground pixels
    foreground_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    
    # Calculate foreground ratio
    foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0
    
    # Count regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)
    
    # Combine with method-specific stats
    stats = {
        "num_regions": method_stats.get("num_regions", num_regions),
        "area_pixels": int(foreground_pixels),
        "foreground_ratio": round(foreground_ratio, 4),
        **method_stats
    }
    
    return stats
