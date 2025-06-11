FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 torch>=2.0.0 matplotlib>=3.7.0 flwr>=1.5.0 gdown>=4.7.0

WORKDIR /app
COPY fl_client_cnn.py .
COPY ../download_dataset.py .
RUN python3 download_dataset.py

CMD ["python3", "fl_client_cnn.py", "client2", "192.168.1.100:8080"]