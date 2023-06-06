import os
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer, SpeechSynthesisOutputFormat
from azure.cognitiveservices.speech.audio import AudioOutputConfig

# AZURE_SPEECH_KEYは環境変数から取得します。
# 空の場合はUsageを表示する
if os.environ.get('AZURE_SPEECH_KEY') is None or os.environ.get('AZURE_SPEECH_REGION') is None:
    print("Usage: AZURE_SPEECH_KEY=<your_key> AZURE_SPEECH_REGION=<your_region> python src/main.py")
    exit(1)

speech_config = SpeechConfig(
    subscription=os.environ.get('AZURE_SPEECH_KEY'), region=os.environ.get('AZURE_SPEECH_REGION'))

# SSML
synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
ssml_string = open("ssml.xml", "r", encoding="utf-8").read()
result = synthesizer.speak_ssml_async(ssml_string).get()

stream = AudioDataStream(result)
stream.save_to_wav_file("voicefile.wav")
