# need to have ffmpeg installed (Works on Google Colab)
# need to install jiwer
# need to have transformers installed
from transformers import pipeline
import os
import pandas as pd
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


filename = r"C:\Users\andrew\Downloads\cv-corpus-13.0-delta-2023-03-09\en\clips" # this has to be the filename of the mainclips audio

data_frames = []
tsv_files = ["other.tsv","invalidated.tsv", "validated.tsv"]
def map_df(tsv_files): # used to create a map of all the audio files

    combined_df = pd.DataFrame()
    for tsv_file in tsv_files:
        df = pd.read_csv(tsv_file, sep='\t')
        data_frames.append(df)
    for d in data_frames:
        combined_df = pd.concat([d, combined_df], axis=0)

    mapper = {row["path"]: row for index, row in combined_df.iterrows()}
    return mapper

mapper = map_df(tsv_files)

def csine(text, reference_text):
    # cosine similarity
    txt = [reference_text, text]
    vct = CountVectorizer()
    vector = vct.fit_transform(txt)
    
    cosine_sim = cosine_similarity(vector)

    similarity = cosine_sim[0][1]

    return similarity

def word_error(text, reference_text):
    wer = evaluate.load("wer")
    return wer.compute(references=[reference_text], predictions=[text])

def multiple_transcriber(filename, numb, type): # transcribe multiple files given filepath of main file

    audio_file_folder = filename

    all_files = os.listdir(audio_file_folder)
    
    all_files.sort() #newest to oldest

    audio_files = [file for file in all_files if file.endswith(type)][:numb]

    for audio_file in audio_files:
        audio_path = os.path.join(audio_file_folder, audio_file)
        transcription = transcriber(audio_path)
        reference_text = mapper[audio_file]["sentence"]
        print(transcription, reference_text)
        print(csine(transcription,reference_text), word_error(transcription,reference_text))

# I HAVE TO WRITE THEM DOWN INTO A TXT FILE

speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

def transcriber(filename): # transcribe audio given the filepath of an audio
    transcribe = speech_recognizer(filename, max_new_tokens=8000)
    return transcribe['text']

multiple_transcriber(filename, 20, ".mp3")

# https://www.hindawi.com/journals/wcmc/2022/4444388/ binary classification
# https://github.com/sharansankar/gender_recognition_svm 
