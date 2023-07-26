import demucs.separate
import os
import shutil


def divide_audio(input_path, file_name):
    if file_name is None:
        file_name = input_path.split("/")[-1].split(".")[0]
        print(f"file_name: {file_name}")
    else:
        input_path = f"downloads/{file_name}.mp3"
    print(f"input_path: {input_path}")

    model_name = "htdemucs_ft"
    demucs.separate.main(
        ["--two-stems", "vocals", "-n", model_name, input_path, "-o", "downloads"])

    # downloadsに作成されたディレクトリの中で最も更新時間が新しいものを取得
    dirs = os.listdir(f"downloads/{model_name}")
    dirs.sort(key=lambda x: os.path.getmtime(f"downloads/{model_name}/" + x))
    latest_dir = dirs[-1]
    print(f"latest_dir: {latest_dir}")
    demucs_save_path = f"/app/downloads/{model_name}/{latest_dir}/vocals.wav"

    # /app/downloads/{file_name}/vocals.wavとして保存する
    dir_name = f"/app/downloads/{file_name}"
    os.makedirs(dir_name, exist_ok=True)
    output_path = f"{dir_name}/vocals.wav"
    os.rename(demucs_save_path, output_path)

    # htdemucs_ftディレクトリを削除する
    shutil.rmtree(f"/app/downloads/{model_name}")

    return output_path
