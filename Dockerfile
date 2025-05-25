# 베이스 이미지 선택 (CUDA와 PyTorch 포함)
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1 \
    libsm6 \
    libglu1-mesa \
    libxi6 \
    libxkbcommon0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Blender 설치
RUN wget https://download.blender.org/release/Blender3.6/blender-3.6.2-linux-x64.tar.xz && \
    tar -xf blender-3.6.2-linux-x64.tar.xz && \
    mv blender-3.6.2-linux-x64 /opt/blender && \
    ln -s /opt/blender/blender /usr/local/bin/blender

# 환경 변수 설정
ENV BLENDER_PATH=/usr/local/bin/blender


# 프로젝트 파일 복사
COPY . /workspace

# 별도의 conda 환경 생성 및 빌드 도구 설치

# conda-forge를 기본 채널로 설정 후 gxx, cxx-compiler, cmake 설치
RUN conda create -n aisystem -c conda-forge -y \
    python=3.12 \
    && conda clean -afy 

RUN /bin/bash -c "source activate aisystem && \
    cd /workspace/shap-e && \
    pip install -e . && \
    pip install -r requirements.txt " 

RUN echo "source activate aisystem" >> ~/.bashrc
