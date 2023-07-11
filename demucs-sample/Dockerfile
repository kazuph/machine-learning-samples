FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt update && apt install -y python3 python3-pip build-essential libssl-dev libffi-dev python3-dev cmake git ffmpeg

WORKDIR /app

RUN pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
# streamlit run app.py & sleep 3 && npx localtunnel --port 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
