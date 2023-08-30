# need to have ffmpeg installed (Works on Google Colab)
# need to have transformers installed
from transformers import pipeline

speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

transcribe = speech_recognizer(r"C:\Users\andrew\Documents\research-ML\topic1-ML\name_andrew.wav", max_new_tokens=8000)
print(transcribe['text'])