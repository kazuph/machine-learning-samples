FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN apt install -y cmake
RUN apt install -y git
RUN apt install -y ffmpeg
RUN apt install -y nvidia-cuda-toolkit

WORKDIR /app

RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 uninstall -y cmake
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install pydub

# RUN pip3 install --no-cache-dir \
#   black \
#   jupyterlab \
#   jupyterlab_code_formatter \
#   jupyterlab-git \
#   lckr-jupyterlab-variableinspector \
#   jupyterlab_widgets \
#   ipywidgets \
#   import-ipynb

# RUN pip install --no-cache-dir \
#     pandas \
#     scipy \
#     scikit-learn \
#     pycaret \
#     matplotlib \
#     japanize_matplotlib \
#     mlxtend \
#     seaborn \
#     plotly \
#     requests \
#     beautifulsoup4 \
#     Pillow \
#     opencv-python

COPY . .
CMD ["python3", "app.py"]
