FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN apt install -y cmake
RUN apt install -y git
RUN apt install -y ffmpeg
RUN apt install -y wget
RUN apt install -y nvidia-cuda-toolkit

WORKDIR /app

RUN pip3 install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 uninstall -y cmake
RUN pip3 install pydub
RUN pip3 install omegaconf

RUN pip install git+https://github.com/huggingface/diffusers -U
RUN pip install -q controlnet-aux transformers accelerate peft gradio==3.50.2
RUN pip install -q https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl

COPY app.py .
# COPY . .
CMD ["python3", "app.py"]
