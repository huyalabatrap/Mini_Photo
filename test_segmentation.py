#!/usr/bin/env python3
"""
Test script to verify segmentation algorithms with sample processing
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from processing.segmentation import run_segmentation

def create_test_image():
    """Create a simple test image with white text on dark background (simulating license plate)"""
    # Create a dark background
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Add white rectangle (plate background)
    cv2.rectangle(img, (50, 25), (250, 75), (240, 240, 240), -1)
    
    # Add black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '92A', (60, 45), font, 0.7, (0, 0, 0), 2)
    cv2.putText(img, '004.46', (60, 65), font, 0.7, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite('test_plate.png', img)
    return 'test_plate.png'

def test_segmentation_methods():
    """Test different segmentation methods"""
    print("🧪 Testing Segmentation Methods\n")
    
    # Create test image
    test_image_path = create_test_image()
    print(f"✅ Created test image: {test_image_path}")
    
    methods = [
        'plate',
        'background', 
        'otsu',
        'kmeans',
        'canny'
    ]
    
    for method in methods:
        print(f"\n🔍 Testing method: {method}")
        
        try:
            result = run_segmentation(test_image_path, method, {}, f"test_{method}")
            
            print(f"   ✅ Success!")
            print(f"   📊 Stats: {result.stats}")
            print(f"   📁 Results saved to: {result.result_path}")
            
            # Check if mask has content
            mask = cv2.imread(str(result.result_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                fg_pixels = np.sum(mask > 0)
                total_pixels = mask.size
                fg_ratio = fg_pixels / total_pixels
                
                if fg_ratio > 0.01:  # At least 1% foreground
                    print(f"   ✅ Mask has content: {fg_ratio:.1%} foreground")
                else:
                    print(f"   ⚠️  Mask mostly empty: {fg_ratio:.1%} foreground")
            else:
                print(f"   ❌ Could not read mask file")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🗑️  Cleaning up test image: {test_image_path}")
    Path(test_image_path).unlink(missing_ok=True)

if __name__ == "__main__":
    test_segmentation_methods()