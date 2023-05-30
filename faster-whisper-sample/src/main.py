# https://zenn.dev/ryoppippi/articles/b66fa477c1c3af
import subprocess
import os
from faster_whisper import WhisperModel

YOUTUBE_ID = "uXCipjbcQfM"  # Youtube ID
AUDIO_FILE_NAME = f"{YOUTUBE_ID}.mp3"


def dl_yt(yt_url):
    # Download audio from Youtube
    subprocess.run(
        f"yt-dlp -x --audio-format mp3 -o {AUDIO_FILE_NAME} {yt_url}", shell=True)


# ファイルがなければ実行
if not os.path.exists(AUDIO_FILE_NAME):
    dl_yt(f"https://youtu.be/{YOUTUBE_ID}")
else:
    print("File already exists")


model_size = "large-v2"

# GPU, FP16で実行
# model=WhisperModel(model_size, device="cuda", compute_type="float16")
# GPU, INT8で実行
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# CPU, FP16で実行
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(AUDIO_FILE_NAME, beam_size=5)

print("Detected language '%s' with probability %f" %
      (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
