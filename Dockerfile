FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     libglib2.0-0 &&     rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .
ENV PORT=7860
EXPOSE 7860
CMD ["bash","-lc","gunicorn -w 1 -k gthread -t 120 -b 0.0.0.0:${PORT} app:app"]
