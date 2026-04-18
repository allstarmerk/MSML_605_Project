FROM python:3.11-slim

WORKDIR /app

# System libs required 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU only PyTorch first 
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install facenet-pytorch and remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir facenet-pytorch && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/      src/
COPY scripts/  scripts/
COPY configs/  configs/

# download FaceNet weights so the container works fully offline at runtime
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2'); print('Weights downloaded.')"

CMD ["python", "scripts/infer.py", "--help"]
