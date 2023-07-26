import os
import shutil
import base64
import subprocess
import contextlib
import wave
import datetime

import torch
import numpy as np
import pandas as pd

from faster_whisper import WhisperModel

from pyannote.core import Segment
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from pydub import AudioSegment


def speech_to_text(input_path, file_name, selected_source_lang, whisper_model, num_speakers):
    if file_name is None:
        file_name = input_path.split("/")[-1].split(".")[0]
        print(f"file_name: {file_name}")
    else:
        input_path = f"downloads/{file_name}/vocals_cut.wav"
    print(f"input_path: {input_path}")

    # output/以下は一度空にする
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs('output', exist_ok=True)

    model = WhisperModel(whisper_model, device="cuda" if torch.cuda.is_available() else "cpu",
                         compute_type="int8_float16")

    # vocals_cutの部分をconvertに変更する
    output_path = input_path.replace("vocals_cut", "convert")

    # 音声認識
    try:
        print("starting conversion to wav")
        os.system(
            f'ffmpeg -y -i "{input_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{output_path}"')

        # Get duration
        with contextlib.closing(wave.open(output_path, 'r')) as f:
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
            output_path, **transcribe_options)

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

    # 話者分離
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
            waveform, sample_rate = audio.crop(output_path, clip)
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

            print(f"audio_file_path: {output_path}")
            print(f"output_file_path: {dir_name}/{file_name}")
            subprocess.run(
                f"ffmpeg -y -i '{output_path}' -ss {segments[i]['start']} -to {segments[i]['end']} -ar 16000 -ac 1 -c:a pcm_s16le {dir_name}/{file_name}",
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

        return df_results_html, save_path

    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))
