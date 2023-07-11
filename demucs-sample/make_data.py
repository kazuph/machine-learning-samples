from pyannote.audio import Pipeline
import whisper
import sys

audio_file = sys.argv[1] if len(sys.argv) > 1 else exit()

# 音声データ取得部
# Pyannote.audioで話者識別の実施
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
diarization = pipeline(audio_file)
speaker_list = [[turn.start, turn.end, speaker, ""]
                for turn, _, speaker in diarization.itertracks(yield_label=True)]

# Whisperで音声認識の実施
model = whisper.load_model("large")
result = model.transcribe(audio_file)


# Whisperの発話範囲から、Pyannote.audioの発話者を特定する関数
def speaker_selector(s, e, t, speaker_dict={}):
    while t <= len(speaker_list):
        if s <= speaker_list[t][0] and e >= speaker_list[t][1]:
            if speaker_list[t][2] in speaker_dict:
                return speaker_dict[speaker_list[t][2]], t
            else:
                return speaker_list[t][2], t
        else:
            if len(speaker_list) >= t-2:
                if e >= speaker_list[t+1][0]:
                    t += 1
                else:
                    return "unknown", t
            else:
                return "unknow", t


# 議事録の出力
speaker_turn = 0
for i in range(len(result['segments'])):
    talk_data = result['segments'][i]
    speaker, speaker_turn = speaker_selector(
        talk_data['start'], talk_data['end'], speaker_turn)
    print("{:.1f}s --- {:.1f}s {} {}".format(
        talk_data['start'], talk_data['end'], speaker, talk_data['text']))
