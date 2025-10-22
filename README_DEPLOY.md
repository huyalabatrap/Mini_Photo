# Deploy Pack (Render/Railway/HF Spaces)

## What to do (Render / Railway)
1. Put **Procfile**, **requirements.txt**, **runtime.txt**, **.python-version** into your repo root (same level as `app.py`).
2. Make sure your `requirements.txt` uses **opencv-python-headless** (not `opencv-python`).  
   If you installed the wrong one locally:
   ```bash
   pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
   pip install "opencv-python-headless>=4.8,<5"
   ```
3. Commit & push to GitHub.

### Render setup
- Create New → Web Service → connect the repo.
- Build Command: `pip install -r requirements.txt`
- Start Command (if asked): `gunicorn app:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:$PORT`
- Render respects `runtime.txt` (python-3.11.9).

### Railway setup
- New Project → Deploy from GitHub repo.
- Ensure env var `PORT` exists (Railway usually injects it).
- Start Command if needed: same as above.
- Railway/mise typically respect `.python-version` → 3.11.9.

## Docker (Hugging Face Spaces or generic)
- Use the provided **Dockerfile**. Spaces Type: *Docker*.
- `requirements.txt` still uses `opencv-python-headless`.
- Spaces will expose `$PORT` (7860 by default).

## Verify locally
```bash
python -c "import cv2; print('cv2 ok', cv2.__version__)"
gunicorn app:app --bind 0.0.0.0:5000
```

## Common failures mapped
- `ImportError: libGL.so.1` → use **opencv-python-headless**.
- `No matching distribution` for OpenCV → pin Python to **3.11** via `runtime.txt` / `.python-version`.
- `ModuleNotFoundError: flask` → check `requirements.txt` and the build step installs it.
- `no Procfile found` → ensure a file named `Procfile` (no extension) at repo root.
- Building from source (`cmake`/`ninja` in logs) → you are installing full OpenCV. Switch to **headless** wheel.
