#!/usr/bin/env python3
"""
Comprehensive test suite for advanced segmentation methods
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from processing.segmentation import run_segmentation

def create_test_images():
    """Create various test images for different segmentation methods"""
    
    test_images = {}
    
    # 1. License plate test image
    plate_img = np.zeros((100, 300, 3), dtype=np.uint8)
    plate_img[:] = (50, 50, 50)
    cv2.rectangle(plate_img, (50, 25), (250, 75), (240, 240, 240), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plate_img, '92A 004.46', (60, 55), font, 0.7, (0, 0, 0), 2)
    cv2.imwrite('test_plate.png', plate_img)
    test_images['plate'] = 'test_plate.png'
    
    # 2. Coins/objects test image
    coins_img = np.zeros((300, 300, 3), dtype=np.uint8)
    coins_img[:] = (100, 100, 100)
    # Draw several circles
    cv2.circle(coins_img, (75, 75), 30, (200, 200, 200), -1)
    cv2.circle(coins_img, (150, 75), 25, (180, 180, 180), -1)
    cv2.circle(coins_img, (225, 75), 28, (190, 190, 190), -1)
    cv2.circle(coins_img, (100, 180), 32, (210, 210, 210), -1)
    cv2.circle(coins_img, (200, 180), 27, (195, 195, 195), -1)
    cv2.circle(coins_img, (150, 240), 30, (185, 185, 185), -1)
    cv2.imwrite('test_coins.png', coins_img)
    test_images['count'] = 'test_coins.png'
    
    # 3. Skin tone test image
    skin_img = np.zeros((200, 200, 3), dtype=np.uint8)
    skin_img[:] = (50, 100, 150)  # Background
    # Create skin-colored region
    skin_color = (120, 160, 200)  # Typical skin tone in BGR
    cv2.ellipse(skin_img, (100, 100), (60, 80), 0, 0, 360, skin_color, -1)
    cv2.imwrite('test_skin.png', skin_img)
    test_images['skin'] = 'test_skin.png'
    
    # 4. Simple foreground/background image
    bg_img = np.zeros((200, 200, 3), dtype=np.uint8)
    bg_img[:] = (30, 30, 30)  # Dark background
    cv2.rectangle(bg_img, (50, 50), (150, 150), (200, 150, 100), -1)  # Colorful foreground
    cv2.imwrite('test_background.png', bg_img)
    test_images['background'] = 'test_background.png'
    
    # 5. Lesion/anomaly test image
    lesion_img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Light background
    # Add dark spots (lesions)
    cv2.circle(lesion_img, (70, 70), 15, (80, 80, 80), -1)
    cv2.circle(lesion_img, (130, 130), 20, (70, 70, 70), -1)
    cv2.ellipse(lesion_img, (150, 80), (12, 18), 30, 0, 360, (90, 90, 90), -1)
    cv2.imwrite('test_lesion.png', lesion_img)
    test_images['lesion'] = 'test_lesion.png'
    
    return test_images

def test_all_methods():
    """Test all segmentation methods with appropriate test images"""
    
    print("🎨 Creating test images...")
    test_images = create_test_images()
    print(f"✅ Created {len(test_images)} test images\n")
    
    # Method configurations
    test_configs = [
        ('plate', test_images.get('plate', 'test_plate.png'), {}, "License Plate Detection"),
        ('count', test_images.get('count', 'test_coins.png'), {}, "Object Counting (Watershed)"),
        ('skin', test_images.get('skin', 'test_skin.png'), {}, "Skin Detection"),
        ('background', test_images.get('background', 'test_background.png'), {}, "Background Removal"),
        ('lesion', test_images.get('lesion', 'test_lesion.png'), {}, "Lesion Detection"),
        ('otsu', test_images.get('plate', 'test_plate.png'), {}, "Otsu Thresholding"),
        ('kmeans', test_images.get('background', 'test_background.png'), {'kmeans_k': 3}, "K-means Clustering"),
        ('canny', test_images.get('coins', 'test_coins.png'), {}, "Canny Edge Detection"),
        ('watershed', test_images.get('coins', 'test_coins.png'), {}, "Watershed Segmentation"),
    ]
    
    results_summary = []
    
    for method, image_path, params, description in test_configs:
        print(f"{'='*60}")
        print(f"🔬 Testing: {description} ({method})")
        print(f"{'='*60}")
        
        try:
            # Run segmentation
            result = run_segmentation(image_path, method, params, f"test_{method}")
            
            # Load and analyze mask
            mask = cv2.imread(str(result.result_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                fg_pixels = np.sum(mask > 0)
                total_pixels = mask.size
                fg_ratio = fg_pixels / total_pixels
                
                # Check quality
                if fg_ratio > 0.01 and fg_ratio < 0.95:
                    quality = "✅ GOOD"
                    status_emoji = "✅"
                elif fg_ratio <= 0.01:
                    quality = "⚠️  EMPTY"
                    status_emoji = "⚠️"
                elif fg_ratio >= 0.95:
                    quality = "⚠️  FULL"
                    status_emoji = "⚠️"
                else:
                    quality = "❓ UNKNOWN"
                    status_emoji = "❓"
                
                print(f"   {status_emoji} Result: {quality}")
                print(f"   📊 Foreground: {fg_ratio:.1%}")
                print(f"   📁 Saved to: {result.result_path}")
                print(f"   📈 Statistics:")
                for key, value in result.stats.items():
                    print(f"      - {key}: {value}")
                
                results_summary.append({
                    'method': description,
                    'quality': quality,
                    'fg_ratio': fg_ratio,
                    'stats': result.stats
                })
            else:
                print(f"   ❌ Could not read mask file")
                results_summary.append({
                    'method': description,
                    'quality': "❌ ERROR",
                    'fg_ratio': 0,
                    'stats': {}
                })
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'method': description,
                'quality': "❌ EXCEPTION",
                'fg_ratio': 0,
                'stats': {}
            })
        
        print()
    
    # Summary report
    print(f"\n{'='*60}")
    print(f"📋 SUMMARY REPORT")
    print(f"{'='*60}\n")
    
    good_count = sum(1 for r in results_summary if "✅" in r['quality'])
    warning_count = sum(1 for r in results_summary if "⚠️" in r['quality'])
    error_count = sum(1 for r in results_summary if "❌" in r['quality'])
    
    print(f"Total tests: {len(results_summary)}")
    print(f"✅ Good results: {good_count}")
    print(f"⚠️  Warnings: {warning_count}")
    print(f"❌ Errors: {error_count}")
    
    print(f"\nDetailed Results:")
    print(f"{'-'*60}")
    for r in results_summary:
        print(f"{r['quality']:15} | {r['method']:30} | {r['fg_ratio']:.1%}")
    
    # Cleanup
    print(f"\n🗑️  Cleaning up test images...")
    for image_path in test_images.values():
        Path(image_path).unlink(missing_ok=True)
    
    print(f"\n{'='*60}")
    if good_count == len(results_summary):
        print("🎉 ALL TESTS PASSED!")
    elif good_count + warning_count == len(results_summary):
        print("⚠️  TESTS COMPLETED WITH WARNINGS")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_all_methods()