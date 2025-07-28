FROM python:3.10-slim

# System dependencies for PyMuPDF and LightGBM
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    gcc \
    g++ \
    python3-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    numpy \
    PyMuPDF \
    joblib \
    scikit-learn \
    lightgbm

CMD ["python", "extract_outline.py"]
