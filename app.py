"""
Flask web application for image segmentation processing.
Provides a modern, user-friendly interface for various segmentation methods.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

import processing.segmentation


# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# Directory paths
UPLOAD_FOLDER = Path('static/uploads')
RESULTS_FOLDER = Path('static/results')


def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def save_job_metadata(job_id: str, method: str, input_path: str, 
                     result_path: str, overlay_path: str, 
                     cutout_path: str, stats: Dict[str, Any]) -> None:
    """Save job metadata to JSON file."""
    metadata = {
        'method': method,
        'input_path': input_path,
        'result_path': result_path,
        'overlay_path': overlay_path,
        'cutout_path': cutout_path,
        'stats': stats
    }
    
    metadata_path = RESULTS_FOLDER / f"{job_id}.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_job_metadata(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job metadata from JSON file."""
    metadata_path = RESULTS_FOLDER / f"{job_id}.json"
    
    if not metadata_path.exists():
        return None
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def parse_advanced_params(form_data: Dict[str, str]) -> Dict[str, Any]:
    """Parse advanced parameters from form data."""
    params = {}
    
    # K-means clusters
    if 'kmeans_k' in form_data and form_data['kmeans_k']:
        try:
            params['kmeans_k'] = int(form_data['kmeans_k'])
        except ValueError:
            pass
    
    # Canny edge detection thresholds
    if 'canny_low' in form_data and form_data['canny_low']:
        try:
            params['canny_low'] = int(form_data['canny_low'])
        except ValueError:
            pass
            
    if 'canny_high' in form_data and form_data['canny_high']:
        try:
            params['canny_high'] = int(form_data['canny_high'])
        except ValueError:
            pass
    
    return params


def validate_file(file: FileStorage) -> Tuple[bool, str]:
    """Validate uploaded file."""
    if not file or not file.filename:
        return False, "Không có file được chọn."
    
    if not is_allowed_file(file.filename):
        return False, f"Định dạng file không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, ""


@app.route('/')
def index() -> str:
    """Render the main page."""
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    """Handle image segmentation request."""
    try:
        # Validate file upload
        if 'image' not in request.files:
            flash('Không có file được tải lên.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['image']
        is_valid, error_msg = validate_file(file)
        
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('index'))
        
        # Get segmentation method
        method = request.form.get('method', 'background')
        if method not in ['background', 'count', 'skin', 'plate', 'lesion', 
                         'otsu', 'kmeans', 'canny', 'watershed']:
            flash('Phương pháp phân đoạn không hợp lệ.', 'error')
            return redirect(url_for('index'))
        
        # Generate job ID and save file
        job_id = str(uuid4())
        filename = secure_filename(file.filename)
        file_extension = Path(filename).suffix
        safe_filename = f"{job_id}{file_extension}"
        
        create_directories()
        input_path = UPLOAD_FOLDER / safe_filename
        file.save(str(input_path))
        
        # Parse advanced parameters
        params = parse_advanced_params(request.form.to_dict())
        
        # Run segmentation
        try:
            result_data = processing.segmentation.run_segmentation(
                str(input_path), method, params, job_id
            )
            
            # Save metadata
            save_job_metadata(
                job_id=job_id,
                method=method,
                input_path=str(input_path).replace('\\', '/'),
                result_path=str(result_data.result_path).replace('\\', '/'),
                overlay_path=str(result_data.overlay_path).replace('\\', '/') if result_data.overlay_path else '',
                cutout_path=str(result_data.cutout_path).replace('\\', '/') if result_data.cutout_path else '',
                stats=result_data.stats
            )
            
            flash('Phân đoạn ảnh thành công!', 'success')
            return redirect(url_for('results', job_id=job_id))
            
        except Exception as e:
            flash(f'Lỗi trong quá trình xử lý: {str(e)}', 'error')
            # Clean up uploaded file on error
            if input_path.exists():
                input_path.unlink()
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Lỗi không mong đợi: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/results/<job_id>')
def results(job_id: str) -> str:
    """Display segmentation results."""
    # Load job metadata
    metadata = load_job_metadata(job_id)
    
    if not metadata:
        flash('Không tìm thấy kết quả phân đoạn.', 'error')
        return redirect(url_for('index'))
    
    # Prepare template variables
    template_vars = {
        'job_id': job_id,
        'method': metadata.get('method', ''),
        'input_path': metadata.get('input_path', ''),
        'result_path': metadata.get('result_path', ''),
        'overlay_path': metadata.get('overlay_path', ''),
        'cutout_path': metadata.get('cutout_path', ''),
        'stats': metadata.get('stats', {})
    }
    
    return render_template('result.html', **template_vars)


def get_filename_from_path(file_path: str) -> str:
    """Extract filename from path, handling both forward and back slashes."""
    return Path(file_path).name


@app.route('/download/<filename>')
def download_file(filename: str):
    """Download result file."""
    try:
        # Clean filename to prevent path traversal
        clean_filename = secure_filename(filename)
        file_path = RESULTS_FOLDER / clean_filename
        
        if not file_path.exists():
            flash('File không tồn tại.', 'error')
            return redirect(url_for('index'))
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        flash(f'Lỗi khi tải file: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File quá lớn. Kích thước tối đa là 10MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    flash('Trang không tìm thấy.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    flash('Lỗi máy chủ nội bộ.', 'error')
    return redirect(url_for('index'))


# Initialize directories on startup
create_directories()


@app.template_filter('get_filename')
def get_filename_filter(path):
    """Template filter to extract filename from path."""
    return Path(path).name


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
