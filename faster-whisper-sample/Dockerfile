FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN apt install -y cmake
RUN apt install -y git
RUN apt install -y ffmpeg
RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 uninstall -y cmake
RUN pip3 install pydub

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]
