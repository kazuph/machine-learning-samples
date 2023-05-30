import whisper

model = whisper.load_model("large")
result = model.transcribe("input.mp3")
print(result["text"])
