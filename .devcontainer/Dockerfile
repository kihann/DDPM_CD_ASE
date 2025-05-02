# Mac에서도 AMD64 에뮬레이션으로 개발 가능
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    curl git vim sudo build-essential \
    && ln -sf python3.12 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip install --no-cache-dir uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 필수 실험 패키지 설치
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    tensorboard

WORKDIR /workspace

