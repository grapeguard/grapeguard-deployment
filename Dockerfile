# Dockerfile (repo root)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps needed for OpenCV and some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake curl libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (better cache)
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --upgrade pip wheel setuptools

# Install PyTorch CPU wheels first (must match Detectron2 wheel)
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2 wheel compatible with torch==2.2 (CPU wheel).
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.2/index.html

# Install project python deps (no torch/detectron2 in this file)
RUN pip install -r /app/backend/requirements.txt

# Copy project
COPY . /app

# Expose port (Render will set PORT)
ENV PORT=10000
ENV MPLCONFIGDIR=/tmp/mpl

EXPOSE ${PORT}

CMD ["bash", "-lc", "gunicorn backend.detectron_server:app -w 1 -k sync -b 0.0.0.0:${PORT} --timeout 120"]
