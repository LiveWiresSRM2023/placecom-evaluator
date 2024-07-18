# Project Name: Audio Transcription and BERT Embedding Similarity Analysis

## Description

This project focuses on extracting audio from videos, transcribing the audio to text, and analyzing the similarity between the transcribed text and a set of predefined keywords using BERT embeddings. The overall goal is to evaluate the relevancy of the transcribed text to specific topics such as self-introduction, public speaking, and other related keywords.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. **Mount Google Drive**:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Install Required Packages**:
   ```bash
   !apt-get install -y ffmpeg
   !pip install transformers
   !pip install datasets
   !pip install torchaudio
   !pip install moviepy
   !pip install SpeechRecognition
   !pip install scikit-learn
   ```

## Usage

1. **Extract Audio from Video**:

   ```python
   from moviepy.editor import VideoFileClip
   import os

   def extract_audio(video_path, audio_path):
       video = VideoFileClip(video_path)
       audio = video.audio
       audio.write_audiofile(audio_path)

   video_path = '/content/drive/MyDrive/Placecom/video1'
   audio_path = '/content/drive/MyDrive/Placecom/video1audio.wav'
   extract_audio(video_path, audio_path)

   if os.path.exists(audio_path):
       print("Audio file created successfully")
   else:
       print("Failed to create audio file")
   ```

2. **Transcribe Audio**:

   ```python
   import speech_recognition as sr

   def transcribe_audio(file_path):
       recognizer = sr.Recognizer()
       audio_file = sr.AudioFile(file_path)

       with audio_file as source:
           audio_data = recognizer.record(source)

       try:
           text = recognizer.recognize_google(audio_data)
           return text
       except sr.RequestError:
           return "API unavailable"
       except sr.UnknownValueError:
           return "Unable to recognize speech"

   audio_file_path = '/content/drive/MyDrive/Placecom/video1audio.wav'
   transcription = transcribe_audio(audio_file_path)
   print(f"Transcribed Text: {transcription}")
   ```

3. **Generate BERT Embeddings and Calculate Similarity**:

   ```python
   from transformers import BertModel, BertTokenizer
   import torch
   from sklearn.metrics.pairwise import cosine_similarity
   import numpy as np

   model_name = 'bert-base-uncased'
   tokenizer = BertTokenizer.from_pretrained(model_name)
   model = BertModel.from_pretrained(model_name)

   def generate_bert_embeddings(text):
       inputs = tokenizer(text, return_tensors="pt")
       with torch.no_grad():
           outputs = model(**inputs)
       embeddings = outputs.last_hidden_state[:, 0, :].numpy()
       return embeddings

   transcribed_text = transcription
   transcribed_embeddings = generate_bert_embeddings(transcribed_text.lower())

   keyword_list = [
       "self introduction",
       "public speaking",
       "presentation skills",
       "communication",
       "confidence",
       "skills",
       "projects",
       "organizing",
       "problem solving",
       "critical thinking",
       "agile",
       "time management",
       "improvement",
       "learnings",
       "motivation"
   ]

   keyword_embeddings = []
   for keyword in keyword_list:
       keyword_embeddings.append(generate_bert_embeddings(keyword.lower()))

   transcribed_embeddings = np.array(transcribed_embeddings)
   keyword_embeddings = np.array(keyword_embeddings)

   def calculate_cosine_similarity(embedding1, embedding2):
       similarity = cosine_similarity(embedding1, embedding2)
       return similarity[0][0]

   scores = []
   for keyword_emb in keyword_embeddings:
       similarity_score = calculate_cosine_similarity(transcribed_embeddings, keyword_emb)
       scores.append(similarity_score)

   for keyword, score in zip(keyword_list, scores):
       print(f"Keyword: {keyword}, Similarity Score: {score}")
   ```

4. **Calculate Overall Score**:

   ```python
   import numpy as np

   def calculate_overall_score(s):
       normalized_scores = [(score - min(s)) / (max(s) - min(s)) for score in s]
       overall_score = np.mean(normalized_scores) * 10
       return overall_score

   overall_score = calculate_overall_score(scores)
   print(f"Overall Score out of 10: {overall_score:.2f}")
   ```

## Features

- Extract audio from video files.
- Transcribe audio to text using Google Speech Recognition.
- Generate BERT embeddings for transcribed text and keywords.
- Calculate cosine similarity between embeddings.
- Compute an overall relevancy score.

## Requirements

- ffmpeg
- transformers
- datasets
- torchaudio
- moviepy
- SpeechRecognition
- scikit-learn

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
