from pydub import AudioSegment
from pydub.silence import split_on_silence


def trim_silence(input_path, file_name, min_silence_len=800, silence_thresh=-60):
    if file_name is None:
        file_name = input_path.split("/")[-1].split(".")[0]
        print(f"file_name: {file_name}")
    else:
        input_path = f"downloads/{file_name}/vocals.wav"
    print(f"input_path: {input_path}")

    sound = AudioSegment.from_wav(input_path)
    chunks = split_on_silence(sound,
                              min_silence_len=min_silence_len,  # 無音と判断する最小の長さ（ミリ秒）
                              silence_thresh=silence_thresh     # 無音と判断する閾値（dB）
                              )

    # 分割したチャンクを結合する
    output = chunks[0]
    for chunk in chunks[1:]:
        output += chunk

    output_path = f"/app/downloads/{file_name}/vocals_cut.wav"
    output.export(output_path, format="wav")

    print(f"Save cut audio: {output_path}")
    return output_path
