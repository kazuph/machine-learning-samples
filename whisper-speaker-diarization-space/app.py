# import whisper
import psutil
import contextlib
import wave
from gpuinfo import GPUInfo
import demucs.separate
from pyannote.core import Segment
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from faster_whisper import WhisperModel
import datetime
import gradio as gr
import pandas as pd
import time
import os
import base64
import subprocess
import shutil
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pydub import AudioSegment
from pydub.silence import split_on_silence

import yt_dlp
import torch

from pyannote.audio import Pipeline
from pydub import AudioSegment

whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
source_languages = {
    "en": "English",
    "ja": "Japanese",
}

source_language_list = [key[0] for key in source_languages.items()]


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


# def get_youtube(video_url):
#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
#     }
#
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(video_url, download=False)
#         abs_video_path = ydl.prepare_filename(info)
#         ydl.process_info(info)
#
#     print("Success download video")
#     print(abs_video_path)
#
#     return abs_video_path

def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloads/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        abs_audio_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
        ydl.process_info(info)

    print("Success download audio")
    print(abs_audio_path)

    return abs_audio_path


def separate_audio(input_path):
    print(f"input_path: {input_path}")
    demucs.separate.main(
        ["--two-stems", "vocals", "-n", "htdemucs_ft", input_path, "-o", "downloads"])

    # downloadsに作成されたディレクトリの中で最も更新時間が新しいものを取得
    dirs = os.listdir("downloads/htdemucs_ft")
    dirs.sort(key=lambda x: os.path.getmtime("downloads/htdemucs_ft/" + x))
    latest_dir = dirs[-1]
    print(f"latest_dir: {latest_dir}")

    return f"/app/downloads/htdemucs_ft/{latest_dir}/vocals.wav"


def cut_audio(input_path):
    sound = AudioSegment.from_wav(input_path)

    chunks = split_on_silence(sound,
                              min_silence_len=600,  # 無音と判断する最小の長さ（ミリ秒）
                              silence_thresh=-40     # 無音と判断する閾値（dB）
                              )

    # 分割したチャンクを結合する
    output = chunks[0]
    for chunk in chunks[1:]:
        output += chunk

    # 結果を保存する

    # downloadsに作成されたディレクトリの中で最も更新時間が新しいものを取得
    dirs = os.listdir("downloads/htdemucs_ft")
    dirs.sort(key=lambda x: os.path.getmtime("downloads/htdemucs_ft/" + x))
    latest_dir = dirs[-1]
    print(f"latest_dir: {latest_dir}")
    output.export(
        f"/app/downloads/htdemucs_ft/{latest_dir}/vocals_cut.wav", format="wav")

    return f"/app/downloads/htdemucs_ft/{latest_dir}/vocals_cut.wav"


def speech_to_text(audio_file_path, selected_source_lang, whisper_model, num_speakers, num_cut_time):

    # output/以下は一度空にする
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs('output', exist_ok=True)

    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.

    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """

    # model = whisper.load_model(whisper_model)
    model = WhisperModel(whisper_model, device="cuda",
                         compute_type="int8_float16")
    # model = WhisperModel(whisper_model, compute_type="int8")
    time_start = time.time()

    print("audio_file_path: ", audio_file_path)
    if (audio_file_path == None):
        raise ValueError("Error no input")

    file_path = audio_file_path
    print(file_path)
    # Read and convert youtube video
    _, file_ending = os.path.splitext(f'{file_path}')
    print(f'file enging is {file_ending}')
    audio_file_path = file_path.replace(file_ending, ".wav")
    print(audio_file_path)

    try:
        print("starting conversion to wav")
        if num_cut_time == 0:
            os.system(
                f'ffmpeg -y -i "{file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_path}"')
        else:
            os.system(
                f'ffmpeg -y -i "{file_path}" -t {num_cut_time} -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file_path}"')

        # Get duration
        with contextlib.closing(wave.open(audio_file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang,
                       beam_size=5, best_of=5, word_timestamps=True)
        transcribe_options = dict(task="transcribe", **options)

        segments = []

        segments_raw, info = model.transcribe(
            audio_file_path, **transcribe_options)

        # Convert back to original openai format
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text

            segments.append(chunk)
            print(
                f"start={chunk['start']:.2f}s end={chunk['end']:.2f}s text={chunk['text']}")

        print("transcribe audio done with fast whisper")
    except Exception as e:
        raise RuntimeError("Error converting video to audio")


def diarization(audio_file_path, segments, num_speakers, duration):

    try:
        embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file_path, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        if num_speakers == 0:
            print("Finding best number of speakers")
            # Find the best number of speakers
            score_num_speakers = {}

            for num_speakers in range(2, 8+1):
                print(f"Number of speakers: {num_speakers}")
                clustering = AgglomerativeClustering(
                    num_speakers).fit(embeddings)
                score = silhouette_score(
                    embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(
                score_num_speakers, key=lambda x: score_num_speakers[x])
            print(
                f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
        else:
            best_num_speaker = num_speakers
            print(f"Number of speakers: {best_num_speaker}")

        # Assign speaker label
        clustering = AgglomerativeClustering(
            best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = f"SPEAKER{(labels[i] + 1):02d}"
            speaker = segments[i]["speaker"]

            # startからendまでの音声を切り出す
            # stdoutは表示しない
            dir_name = f"output/{speaker}"
            os.makedirs(dir_name, exist_ok=True)

            file_name = f"{segments[i]['start']:04.2f}_{segments[i]['end']:04.2f}_{abs(segments[i]['end'] - segments[i]['start']):02.1f}.wav"

            print(f"audio_file_path: {audio_file_path}")
            print(f"output_file_path: {dir_name}/{file_name}")
            subprocess.run(
                f"ffmpeg -y -i '{audio_file_path}' -ss {segments[i]['start']} -to {segments[i]['end']} -ar 16000 -ac 1 -c:a pcm_s16le {dir_name}/{file_name}",
                shell=True,
                stdout=subprocess.DEVNULL,
            )

            # wavをbase64に変換して保存
            segments[i]["audio"] = "data:audio/wav;base64," + base64.b64encode(
                open(f"{dir_name}/{file_name}", "rb").read()).decode("utf-8")

        # Make output
        objects = {
            'Start': [],
            'End': [],
            'Speaker': [],
            'Text': [],
            'Audio': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            objects['Start'].append(str(convert_time(segment["start"])))
            objects['End'].append(str(convert_time(segment["end"])))
            objects['Speaker'].append(segment["speaker"])
            objects['Text'].append(segment["text"])
            objects['Audio'].append(
                f"<audio controls src='{segment['audio']}' ></audio>")

        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.*
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """
        save_path = "output/transcript_result.csv"
        df_results = pd.DataFrame(objects)
        df_results_html = df_results.to_html(escape=False, render_links=False,
                                             index=False, header=False)
        df_results.to_csv(save_path)

        # output以下のディレクトリを列挙
        output_dirs = [f for f in os.listdir(
            "output") if os.path.isdir(os.path.join("output", f))]

        for output_dir in output_dirs:
            output_dir = "output/" + output_dir
            # すべてのwavファイルを連結するための空のAudioSegmentオブジェクトを作成します
            combined = AudioSegment.empty()

            # 指定されたディレクトリ内のすべてのファイルをループします
            filenames = os.listdir(output_dir)
            filenames.sort()
            for filename in filenames:
                if filename.endswith(".wav"):
                    print(f"filename: {filename}")
                    # wavファイルを見つけたら、それをAudioSegmentオブジェクトに読み込みます
                    audio = AudioSegment.from_wav(
                        os.path.join(output_dir, filename))
                    # その音声を連結します
                    combined += audio

            # 連結した音声ファイルを保存します
            combined.export(
                f"{output_dir}/combined.wav", format="wav")

            # 連結した音声ファイルの長さをミリ秒単位で取得し、秒単位に変換します
            length_seconds = len(combined) / 1000

            print(
                f"{output_dir}/combined.wav: {length_seconds}s")

        return df_results_html, system_info, save_path

    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
audio_in = gr.Audio(label="Audio file", mirror_webcam=False, type="filepath")
separated_audio_in = gr.Audio(
    label="音源分離済みAudio file", mirror_webcam=False, type="filepath")
cut_audio_in = gr.Audio(
    label="無音カット済みAudio file", mirror_webcam=False, type="filepath")

df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
memory = psutil.virtual_memory()
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value",
                                   value="ja", label="Spoken language", interactive=True)
selected_whisper_model = gr.Dropdown(
    choices=whisper_models, type="value", value="large-v2", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(
    precision=0, value=0, label="Input number of speakers for better results. If value=0, model will automatic find the best number of speakers", interactive=True)
number_cut_time = gr.Number(
    precision=0, value=0, label="Input number of audio cut time. If value=0, not cut audio", interactive=True)
system_info = gr.Markdown(
    f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
download_transcript = gr.File(label="Download transcript")
transcription_df = gr.DataFrame(value=df_init, label="Transcription dataframe", row_count=(
    0, "dynamic"), max_rows=10, wrap=True, overflow_row_behaviour='paginate')
transcription_html = gr.HTML(label="Transcription html", html_content="")
title = "Whisper speaker diarization"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
    gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Whisper speaker diarization</h1>
            </div>
        ''')

    # YoutubeのURLから音声をダウンロードする
    with gr.Column():
        gr.Markdown('''
            ## Youtubeの動画のダウンロード（音声のみ）
            ''')
        with gr.Column():
            gr.Markdown('''
                    ### You can test by following examples:
                    ''')
            examples = gr.Examples(examples=["https://www.youtube.com/watch?v=j7BfEzAFuYc&t=32s",
                                             "https://www.youtube.com/watch?v=-UX0X45sYe4",
                                             "https://www.youtube.com/watch?v=7minSgqi-Gw"],
                                   label="Examples", inputs=[youtube_url_in])

        with gr.Column():
            youtube_url_in.render()
            download_youtube_btn = gr.Button("Download Youtube video")
            download_youtube_btn.click(get_youtube, [youtube_url_in], [
                audio_in])

    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 音声のみの分離
                ''')
            audio_in.render()
            separate_btn = gr.Button("Separate audio")
            separate_btn.click(separate_audio, [audio_in], [
                               separated_audio_in])
    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 無音部分のカット
                ''')
            separated_audio_in.render()
            cut_btn = gr.Button("cut audio")
            cut_btn.click(cut_audio, [separated_audio_in], [
                cut_audio_in])
    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 音声認識
                ''')
            cut_audio_in.render()
            with gr.Column():
                selected_source_lang.render()
                selected_whisper_model.render()
                transcribe_btn = gr.Button(
                    "Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text,
                                     [cut_audio_in, selected_source_lang,
                                         selected_whisper_model, number_speakers, number_cut_time],
                                     # [transcription_df, system_info,
                                     [transcription_html, system_info,
                                         download_transcript]
                                     )

    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 話者分離
                ''')
            number_speakers.render()
            number_cut_time.render()

    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 結果
                ''')
            download_transcript.render()
            # transcription_df.render()
            transcription_html.render()
            system_info.render()
            gr.Markdown('''<center><img src='https://visitor-badge.glitch.me/badge?page_id=WhisperDiarizationSpeakers' alt='visitor badge'><a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg' alt='License: Apache 2.0'></center>''')

demo.queue().launch(debug=True, server_port=7860, server_name="0.0.0.0")
