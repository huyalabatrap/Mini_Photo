#!/usr/bin/env python3
import os
import io
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from image_processing import apply_filter, ensure_odd, allowed_file
from presets import run_pipeline, analyze_image, suggest_preset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.environ.get('UPLOAD_DIR', os.path.join(BASE_DIR, 'uploads'))
OUTPUT_FOLDER = os.environ.get('OUTPUT_DIR', os.path.join(BASE_DIR, 'outputs'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB

def save_pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert('RGB'))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def load_file_to_cv2(file_storage):
    pil_img = Image.open(file_storage.stream).convert('RGB')
    bgr = save_pil_to_cv2(pil_img)
    return bgr

def image_histogram_png(bgr):
    """Return base64 PNG of grayscale histogram."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fig = plt.figure(figsize=(3.2, 2.4), dpi=120)
    plt.hist(gray.ravel(), bins=256, range=(0,255))
    plt.title("Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{b64}"

import base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    Accepts:
      - image: uploaded file (optional if original_path provided)
      - group: blur|sharpen|edge|advanced
      - method: see UI
      - kernel_size, sigma, threshold, threshold1, threshold2, amount, alpha
      - Advanced: gamma, c, r1, s1, r2, s2, clip_limit, tile_grid_size, diameter, sigma_color, sigma_space
      - original_path
      - show_hist: '1' to include base64 histograms of before/after in JSON
      - want_json: default '1'
    Returns JSON with URLs (and optionally histograms).
    """
    want_json = request.form.get('want_json', '1') == '1'
    show_hist = request.form.get('show_hist', '0') == '1'

    # Retrieve or load original
    uploaded = request.files.get('image')
    original_rel = request.form.get('original_path', '').strip()
    original_abs = None

    if uploaded and uploaded.filename and allowed_file(uploaded.filename):
        filename = secure_filename(uploaded.filename)
        name, ext = os.path.splitext(filename)
        unique = f"{name}_{uuid.uuid4().hex[:8]}{ext or '.png'}"
        original_abs = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        uploaded.save(original_abs)
        original_rel = f"/uploads/{unique}"
    elif original_rel:
        original_abs = os.path.join(BASE_DIR, original_rel.lstrip('/'))
        if not os.path.exists(original_abs):
            return jsonify({'error': 'Original image not found on server. Please upload again.'}), 400
    else:
        return jsonify({'error': 'No image provided.'}), 400

    orig_img = cv2.imread(original_abs, cv2.IMREAD_COLOR)
    if orig_img is None:
        with open(original_abs, 'rb') as f:
            pil_img = Image.open(f).convert('RGB')
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Parse parameters
    group = request.form.get('group', 'blur')
    method = request.form.get('method', 'gaussian')

    def fnum(key, default):
        try:
            return float(request.form.get(key, str(default)) or default)
        except Exception:
            return default

    def fint(key, default):
        try:
            return int(float(request.form.get(key, str(default)) or default))
        except Exception:
            return default

    params = dict(
        kernel_size=fint('kernel_size', 3),
        sigma=fnum('sigma', 1.0),
        threshold=fnum('threshold', 128),
        threshold1=fnum('threshold1', 100),
        threshold2=fnum('threshold2', 200),
        amount=fnum('amount', 1.5),
        alpha=fnum('alpha', 1.0),
        gamma=fnum('gamma', 1.0),
        c=fnum('c', 1.0),
        r1=fnum('r1', 50),
        s1=fnum('s1', 0),
        r2=fnum('r2', 200),
        s2=fnum('s2', 255),
        clip_limit=fnum('clip_limit', 2.0),
        tile_grid_size=fint('tile_grid_size', 8),
        diameter=fint('diameter', 9),
        sigma_color=fnum('sigma_color', 75),
        sigma_space=fnum('sigma_space', 75),
    )

    # Process
    processed = apply_filter(orig_img, group=group, method=method, **params)

    # Save output
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_name = f"output_{method}_{ts}_{uuid.uuid4().hex[:6]}.png"
    out_abs = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
    cv2.imwrite(out_abs, processed)
    processed_rel = f"/outputs/{out_name}"

    # Save metadata
    meta = {
        "timestamp_utc": ts,
        "original_path": original_rel,
        "processed_path": processed_rel,
        "group": group,
        "method": method,
        "params": params
    }
    meta_name = out_name.replace('.png', '.json')
    meta_abs = os.path.join(app.config['OUTPUT_FOLDER'], meta_name)
    with open(meta_abs, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    resp = {
        "original_url": original_rel,
        "processed_url": processed_rel,
        "metadata_url": f"/outputs/{meta_name}",
        "message": "OK"
    }

    if show_hist:
        try:
            resp["hist_before"] = image_histogram_png(orig_img)
            resp["hist_after"] = image_histogram_png(processed)
        except Exception as e:
            resp["hist_error"] = str(e)

    if want_json:
        return jsonify(resp)

    # direct image response (rare)
    with open(out_abs, 'rb') as f:
        data = f.read()
    return send_file(io.BytesIO(data), mimetype='image/png', as_attachment=False, download_name=out_name)

@app.route('/uploads/<path:fname>')
def serve_upload(fname):
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    return send_file(path, as_attachment=False)

@app.route('/outputs/<path:fname>')
def serve_output(fname):
    path = os.path.join(app.config['OUTPUT_FOLDER'], fname)
    return send_file(path, as_attachment=False)


@app.route('/preset', methods=['POST'])
def preset():
    """
    Apply a combo preset pipeline.
    Accepts:
      - image or original_path
      - preset: one of
        sharpen_blurry, enhance_low_light, denoise_smooth, boost_contrast, artistic_hdr, medical_clarity
      - save_intermediates: '1' to save and return stage images
      - show_hist: '1' to include histograms for before/after
      - want_json: default '1'
    """
    want_json = request.form.get('want_json', '1') == '1'
    show_hist = request.form.get('show_hist', '0') == '1'
    save_inter = request.form.get('save_intermediates', '1') == '1'
    preset_name = request.form.get('preset', 'sharpen_blurry')

    uploaded = request.files.get('image')
    original_rel = request.form.get('original_path', '').strip()
    original_abs = None

    if uploaded and uploaded.filename and allowed_file(uploaded.filename):
        from werkzeug.utils import secure_filename
        filename = secure_filename(uploaded.filename)
        name, ext = os.path.splitext(filename)
        unique = f"{name}_{uuid.uuid4().hex[:8]}{ext or '.png'}"
        original_abs = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        uploaded.save(original_abs)
        original_rel = f"/uploads/{unique}"
    elif original_rel:
        original_abs = os.path.join(BASE_DIR, original_rel.lstrip('/'))
        if not os.path.exists(original_abs):
            return jsonify({'error': 'Original image not found on server. Please upload again.'}), 400
    else:
        return jsonify({'error': 'No image provided.'}), 400

    orig_img = cv2.imread(original_abs, cv2.IMREAD_COLOR)
    if orig_img is None:
        with open(original_abs, 'rb') as f:
            pil_img = Image.open(f).convert('RGB')
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    final_bgr, stages, filters_used = run_pipeline(orig_img, preset_name)

    # Save final
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_name = f"preset_{preset_name}_{ts}_{uuid.uuid4().hex[:6]}.png"
    out_abs = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
    cv2.imwrite(out_abs, final_bgr)
    processed_rel = f"/outputs/{out_name}"

    # Save stages
    stage_urls = []
    if save_inter:
        for i, (label, img) in enumerate(stages):
            sname = f"{out_name[:-4]}_stage{i+1}.png"
            sabs = os.path.join(app.config['OUTPUT_FOLDER'], sname)
            cv2.imwrite(sabs, img)
            stage_urls.append({"label": label, "url": f"/outputs/{sname}"})

    # Metadata
    meta = {
        "timestamp_utc": ts,
        "original_path": original_rel,
        "processed_path": processed_rel,
        "preset": preset_name,
        "filters_used": filters_used
    }
    meta_name = out_name.replace('.png', '.json')
    meta_abs = os.path.join(app.config['OUTPUT_FOLDER'], meta_name)
    with open(meta_abs, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    resp = {
        "original_url": original_rel,
        "processed_url": processed_rel,
        "metadata_url": f"/outputs/{meta_name}",
        "stages": stage_urls,
        "message": "OK"
    }

    if show_hist:
        try:
            resp["hist_before"] = image_histogram_png(orig_img)
            resp["hist_after"] = image_histogram_png(final_bgr)
        except Exception as e:
            resp["hist_error"] = str(e)

    if want_json:
        return jsonify(resp)

    with open(out_abs, 'rb') as f:
        data = f.read()
    return send_file(io.BytesIO(data), mimetype='image/png', as_attachment=False, download_name=out_name)

@app.route('/suggest', methods=['POST'])
def suggest():
    """
    Analyze an image and suggest a preset.
    Accepts: image or original_path
    Returns: {recommended_preset, metrics}
    """
    uploaded = request.files.get('image')
    original_rel = request.form.get('original_path', '').strip()
    original_abs = None

    if uploaded and uploaded.filename and allowed_file(uploaded.filename):
        from werkzeug.utils import secure_filename
        filename = secure_filename(uploaded.filename)
        name, ext = os.path.splitext(filename)
        unique = f"{name}_{uuid.uuid4().hex[:8]}{ext or '.png'}"
        original_abs = os.path.join(app.config['UPLOAD_FOLDER'], unique)
        uploaded.save(original_abs)
        original_rel = f"/uploads/{unique}"
    elif original_rel:
        original_abs = os.path.join(BASE_DIR, original_rel.lstrip('/'))
        if not os.path.exists(original_abs):
            return jsonify({'error': 'Original image not found on server. Please upload again.'}), 400
    else:
        return jsonify({'error': 'No image provided.'}), 400

    img = cv2.imread(original_abs, cv2.IMREAD_COLOR)
    if img is None:
        with open(original_abs, 'rb') as f:
            pil_img = Image.open(f).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    metrics = analyze_image(img)
    reco = suggest_preset(metrics)
    return jsonify({"recommended_preset": reco, "metrics": metrics, "original_url": original_rel})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
