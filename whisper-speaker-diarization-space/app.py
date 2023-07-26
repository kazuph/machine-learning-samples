import gradio as gr
import pandas as pd

from youtube_download import get_youtube
from divide_audio_sources import divide_audio
from pydub_silence import trim_silence
from whisper import speech_to_text


def load_audio(file_path):
    return file_path


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
source_languages = {
    "en": "English",
    "ja": "Japanese",
}

source_language_list = [key[0] for key in source_languages.items()]

youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
file_name_in = gr.Textbox(label="File Name", lines=1, interactive=True)
file_path_in = gr.Textbox(label="File Path", lines=1, interactive=True)

audio_in = gr.Audio(label="Audio file", mirror_webcam=False, type="filepath")
separated_audio_in = gr.Audio(
    label="音源分離済みAudio file", mirror_webcam=False, type="filepath")
cut_audio_in = gr.Audio(
    label="無音カット済みAudio file", mirror_webcam=False, type="filepath")

df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value",
                                   value="ja", label="Spoken language", interactive=True)
selected_whisper_model = gr.Dropdown(
    choices=whisper_models, type="value", value="large-v2", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(
    precision=0, value=0, label="Input number of speakers for better results. If value=0, model will automatic find the best number of speakers", interactive=True)
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
                audio_in, file_path_in, file_name_in])
            file_name_in.render()
            file_path_in.render()
            load_audio_btn = gr.Button("Load Audio from File Path")
            load_audio_btn.click(load_audio, [file_path_in], [audio_in])

    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 音声のみの分離
                ''')
            audio_in.render()
            separate_btn = gr.Button("Separate audio")
            separate_btn.click(divide_audio, [audio_in, file_name_in], [
                               separated_audio_in])
    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 無音部分のカット
                ''')
            separated_audio_in.render()
            cut_btn = gr.Button("cut audio")
            cut_btn.click(trim_silence, [separated_audio_in, file_name_in], [
                cut_audio_in])
    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 音声認識と話者分離
                ''')
            cut_audio_in.render()
            with gr.Column():
                selected_source_lang.render()
                selected_whisper_model.render()
                number_speakers.render()
                transcribe_btn = gr.Button(
                    "Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text,
                                     [cut_audio_in, file_name_in,
                                         selected_source_lang, selected_whisper_model, number_speakers],
                                     [transcription_html, download_transcript]
                                     )
                download_transcript.render()

    with gr.Column():
        with gr.Column():
            gr.Markdown('''
                ## 結果
                ''')
            # transcription_df.render()
            transcription_html.render()
            gr.Markdown('''<center><img src='https://visitor-badge.glitch.me/badge?page_id=WhisperDiarizationSpeakers' alt='visitor badge'><a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg' alt='License: Apache 2.0'></center>''')

demo.queue().launch(debug=True, server_port=7860, server_name="0.0.0.0")
