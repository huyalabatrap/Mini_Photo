#!/usr/bin/env python3
"""
Test script to verify the segmentation app functionality
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import flask
        print(f"✅ Flask {flask.__version__} imported successfully")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__} imported successfully")
        
        from PIL import Image
        print(f"✅ Pillow imported successfully")
        
        # Test our modules
        from processing.segmentation import SegmentationResult, run_segmentation
        print("✅ Segmentation module imported successfully")
        
        import app
        print("✅ Flask app module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    required_dirs = [
        "static/uploads",
        "static/results", 
        "static/css",
        "static/js",
        "templates",
        "processing"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ Directory {dir_path} exists")
        else:
            print(f"❌ Directory {dir_path} missing")
            all_exist = False
    
    return all_exist

def test_files():
    """Test if required files exist"""
    required_files = [
        "app.py",
        "requirements.txt",
        ".gitignore",
        "processing/__init__.py",
        "processing/segmentation.py",
        "templates/_base.html",
        "templates/index.html", 
        "templates/result.html",
        "static/css/style.css",
        "static/js/main.js",
        "static/uploads/.gitkeep",
        "static/results/.gitkeep"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ File {file_path} exists")
        else:
            print(f"❌ File {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🔍 Testing Segmentation App Setup\n")
    
    print("📦 Testing Imports:")
    imports_ok = test_imports()
    print()
    
    print("📁 Testing Directories:")
    dirs_ok = test_directories()
    print()
    
    print("📄 Testing Files:")
    files_ok = test_files()
    print()
    
    if imports_ok and dirs_ok and files_ok:
        print("🎉 All tests passed! App is ready to run.")
        print("\nTo start the app:")
        print("python app.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())