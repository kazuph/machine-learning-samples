# https://zenn.dev/ryoppippi/articles/b66fa477c1c3af
import subprocess
import os
from faster_whisper import WhisperModel


def dl_yt(yt_url, output_file):
    # Download audio from Youtube
    subprocess.run(
        f"yt-dlp -x --audio-format mp3 -o {output_file} {yt_url}", shell=True)


def transcribe(audio_file, word_timestamps=False):
    # 言語モデルのサイズ
    model_size = "large-v2"

    # GPU, FP16で実行
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # GPU, INT8で実行
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # CPU, FP16で実行
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        audio_file, beam_size=5, word_timestamps=word_timestamps)

    print("Detected language '%s' with probability %f" %
          (info.language, info.language_probability))

    for segment in segments:
        print("[%.4fs -> %.4fs] %s" %
              (segment.start, segment.end, segment.text))


# このファイルが直接実行された場合のみ実行
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("-i", "--input", type=str, default=None)
    # wordはbool型で指定する
    parser.add_argument("--word", action="store_true")
    args = parser.parse_args()

    if args.url is None and args.input is None:
        print("Please specify either --url or --input")

    # outputsフォルダがなければ作成
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    input_file = ""

    # urlがあればダウンロードする
    if args.url is not None:
        yt_id = args.url.split("v=")[-1]
        input_file = f"outputs/{yt_id}.mp3"
        dl_yt(args.url)

    if args.input is not None:
        input_file = args.input

    transcribe(input_file, args.word)
