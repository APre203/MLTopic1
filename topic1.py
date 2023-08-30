# need to have ffmpeg installed (Works on Google Colab)
# need to have transformers installed
from transformers import pipeline
import os

filename = r"C:\Users\andrew\Downloads\cv-corpus-13.0-delta-2023-03-09\en\clips" # this has to be the filename of the mainclips audio

def multiple_transcriber(filename, numb, type): # transcribe multiple files given filepath of main file

    audio_file_folder = filename

    all_files = os.listdir(audio_file_folder)
    audio_files = [file for file in all_files if file.endswith(type)][:numb]

    for audio_file in audio_files:
        audio_path = os.path.join(audio_file_folder, audio_file)
        print(transcriber(audio_path))


speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

def transcriber(filename): # transcribe audio given the filepath of an audio
    transcribe = speech_recognizer(filename, max_new_tokens=8000)
    return transcribe['text']

multiple_transcriber(filename, 20, ".mp3")