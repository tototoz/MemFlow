# LIBERO + Diffusion Policy Docker Environment
# CUDA 11.3 + PyTorch 1.11 + MuJoCo

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.8 python3-pip python3.8-dev \
    # 基础工具
    git wget curl vim \
    # 编译工具
    build-essential cmake \
    # mujoco-py 编译依赖
    libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev \
    libglew-dev libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.8 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "typing-extensions<4.12" "cryptography<42" && \
    pip install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV MUJOCO_EGL_DEVICE_ID=0
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# 默认命令
CMD ["/bin/bash"]
