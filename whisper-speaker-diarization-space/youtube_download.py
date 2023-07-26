import yt_dlp


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

    file_name = abs_audio_path.split("/")[-1].split(".")[0]

    return abs_audio_path, abs_audio_path, file_name
