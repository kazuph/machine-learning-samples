import sys
import os
import azure.cognitiveservices.speech as speechsdk

# AZURE_SPEECH_KEYは環境変数から取得します。
# 空の場合はUsageを表示する
if os.environ.get('AZURE_SPEECH_KEY') is None or os.environ.get('AZURE_SPEECH_REGION') is None:
    print("Usage: AZURE_SPEECH_KEY=<your_key> AZURE_SPEECH_REGION=<your_region> python src/speak.py")
    exit(1)

# 環境変数からサブスクリプションキーとリージョンを取得します。
speech_config = speechsdk.SpeechConfig(
    subscription=os.environ.get('AZURE_SPEECH_KEY'),
    region=os.environ.get('AZURE_SPEECH_REGION'))

# デフォルトのスピーカーを使用するためのオーディオ設定を作成します。
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# 使用する音声の言語を設定します。
speech_config.speech_synthesis_voice_name = 'ja-JP-AoiNeural'

# SpeechSynthesizerオブジェクトを作成します。
speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config, audio_config=audio_config)

# ユーザーからテキストを取得し、それを音声に変換します。
# コマンドラインから --text で指定された場合
if len(sys.argv) > 1 and sys.argv[1] == '--text':
    text = sys.argv[2]
else:
    print("Enter some text that you want to speak >")
    text = input()
speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

# 結果を確認します。
if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesized for text [{}]".format(text))
elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
    cancellation_details = speech_synthesis_result.cancellation_details
    print("Speech synthesis canceled: {}".format(cancellation_details.reason))
    if cancellation_details.reason == speechsdk.CancellationReason.Error:
        if cancellation_details.error_details:
            print("Error details: {}".format(
                cancellation_details.error_details))
    print("Did you set the speech resource key and region values?")
