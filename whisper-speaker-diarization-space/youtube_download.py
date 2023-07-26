import yt_dlp
import os


def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '/app/downloads/%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        abs_audio_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
        ydl.process_info(info)

    print("Success download audio")
    print(abs_audio_path)

    # ファイル名にする上で邪魔な文字を削除する
    audio_path = abs_audio_path.replace(" ", "_")
    os.rename(abs_audio_path, audio_path)

    file_name = audio_path.split("/")[-1].split(".")[0]

    return audio_path, audio_path, file_name
