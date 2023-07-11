import streamlit as st
from yt_dlp import YoutubeDL
import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import os
import re
import subprocess
root = '/app'
model = "htdemucs"
# we will look for all those file types.
extensions = ["mp3", "wav", "ogg", "flac"]
two_stems = None   # only separate one stems from the rest, for instance
# two_stems = "vocals"

# Options for the output audio.
mp3 = True
mp3_rate = 320
float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
int24 = False    # output as int24 wavs, unused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!

in_path = root + '/demucs/'
out_path = root + '/demucs_separated/'


def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out


def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()


def separate(inp=None, outp=None):
    inp = inp or in_path
    outp = outp or out_path
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]
    files = [str(f) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {in_path}")
        return
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")


def playsound_ui():

    out_path2 = out_path + 'htdemucs/'
    folder_name = st.selectbox("音源分離した曲を選択", os.listdir(out_path2))
    folder_path = out_path2 + folder_name
    # 音声ファイルが格納されているフォルダが存在する場合
    if os.path.isdir(folder_path):

        # 音声ファイルを取得する
        audio_files = [file for file in os.listdir(
            folder_path) if file.endswith(".mp3")]

        # 音声ファイルが存在する場合
        if audio_files:
            # 音声ファイルを表示する
            for i, file in enumerate(audio_files):
                with open(os.path.join(folder_path, file), "rb") as f:
                    audio_bytes = f.read()

                st.subheader(f"{os.path.splitext(file)[0]}")
                st.audio(audio_bytes, format="audio/mp3")

        # 音声ファイルが存在しない場合
        else:
            st.warning("音声ファイルが見つかりませんでした。")

    # 音声ファイルが格納されているフォルダが存在しない場合
    else:
        st.warning("フォルダが見つかりませんでした。")


def download_soundfile(url):

    ydl_opts = {'format': 'bestaudio',
                'outtmpl': root + '/demucs/%(title)s.mp3'}
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        res = ydl.extract_info(url, download=False)
    st.success("音源ファイルのダウンロードが完了しました。")
    return res['title']


def main():
    input_url = ''

    st.title("音源分離アプリ")
    st.empty()
    st.subheader("音源分離の実行")
    input_url = st.text_input("Youtube の URL を入力してください")

    state_button = st.button("実行")
    if state_button and re.search('youtube.com/watch', input_url) == None and input_url != "":
        st.error("1つの曲にしてください、もしくはURLが誤っています")
    elif state_button and input_url != '':
        st.write("入力されたURL：", input_url)
        title = download_soundfile(input_url)

        with st.spinner("音源分離を実行中...(曲の長さによって実行時間が変わります)"):
            separate()
        st.success("音源分離が完了しました")
        subprocess.run(f'rm {in_path}*.mp3', shell=True)

    elif state_button and input_url == '':
        st.error("URLが入力されていません")

    st.write(" ")
    if os.path.isdir(out_path):
        st.subheader("音源分離された音源の再生")
        playsound_ui()


if __name__ == "__main__":
    main()
