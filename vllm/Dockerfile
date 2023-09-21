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

RUN apt install -y libatlas-base-dev gfortran
RUN pip3 install --upgrade pip \
&&  pip3 install --no-cache-dir \
    black \
    jupyterlab \
    jupyterlab_code_formatter \
    jupyterlab-git \
    lckr-jupyterlab-variableinspector \
    jupyterlab_widgets \
    ipywidgets \
    import-ipynb

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -e . 

# COPY . .
CMD ["python3", "main.py"]
