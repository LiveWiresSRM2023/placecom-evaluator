# -*- coding: utf-8 -*-
"""Overall_scores_pipeline.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y_zFnssZYIZ8O__X9Ic170c4O_MAK6cd
"""

import pandas as pd

df = pd.read_excel("/content/transcripts.xlsx")

df.head()

key = pd.read_excel("/content/keywords.xlsx")

keywords = key['Keywords'].to_list()

# prompt: Using dataframe df: use this transcriipt colomuns to check the relevance of information realted to a particulat list of words using embedinngs and cosine similarity and givethe score out of 10 for  each transcripts in new column usinbg transformers bge-3-small-en-v1.5 model

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Assuming you have a list of relevant words
relevant_words = keywords

# Load the BGE-small-en-v1.5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Function to calculate relevance score
def calculate_relevance_score(transcript, relevant_words):
    # Tokenize the transcript and relevant words
    # Moved 'return_tokens' argument to __call__ method
    transcript_tokens = tokenizer(transcript, return_tensors='pt', truncation=True, padding=True)
    relevant_word_tokens = tokenizer(relevant_words, return_tensors='pt', truncation=True, padding=True)

    # Get embeddings for the transcript and relevant words
    with torch.no_grad():
        transcript_embedding = model(**transcript_tokens).pooler_output
        relevant_word_embeddings = model(**relevant_word_tokens).pooler_output

    # Calculate the average cosine similarity between the transcript and relevant words
    similarities = [cosine_similarity(transcript_embedding.numpy().flatten(), word_embedding.numpy().flatten()) for word_embedding in relevant_word_embeddings]
    average_similarity = np.mean(similarities)

    # Scale the similarity to a score out of 10
    score = average_similarity * 10
    return score

# Calculate relevance scores for each transcript
df['Relevance Score'] = df['Transcript'].apply(lambda x: calculate_relevance_score(x, relevant_words))

print(df)

df.drop(columns= ['Score'])

df.to_excel("/content/relevance_scores.xlsx")

import requests

def evaluate_grammar(text):
    """
    Evaluates the grammar of a given text using LanguageTool API.

    Args:
        text (str): The text to evaluate.

    Returns:
        float: A grammar score out of 10.
    """
    url = "https://api.languagetool.org/v2/check"
    data = {
        'text': text,
        'language': 'en-US'
    }
    response = requests.post(url, data=data)

    # Handle potential API errors
    if response.status_code != 200:
        print("Error communicating with LanguageTool API.")
        return None  # Or handle the error in a way suitable for your application

    matches = response.json().get('matches', [])
    errors = len(matches)
    total_words = len(text.split())

    # Calculate grammar score (avoiding potential division by zero)
    if total_words > 0:
        grammar_score = max(0, 10 - (errors / total_words) * 10)
    else:
        grammar_score = 10  #  Assume perfect score if no words

    return grammar_score

# Example usage
text_to_evaluate = "This is a sentence with some grammer mistakes."
score = evaluate_grammar(text_to_evaluate)

if score is not None:
    print(f"Grammar score: {score:.2f} out of 10")

import requests

# ... (Your existing code for evaluate_grammar function) ...

# Calculate grammar scores for each transcript
df['Grammar Score'] = df['Transcript'].apply(evaluate_grammar)

print(df[['Transcript', 'Grammar Score']])



from transformers import pipeline

def evaluate_emotion(text):
    """
    Evaluates the emotion of a given text and returns the most probable emotion and its score.

    Args:
        text (str): The text to evaluate.

    Returns:
        tuple: A tuple containing the most probable emotion label and its score.
    """
    emotion_pipeline = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')
    emotions = emotion_pipeline(text)

    if emotions:
        # Get the emotion with the highest score
        top_emotion = max(emotions, key=lambda x: x['score'])
        return top_emotion['label'], top_emotion['score']
    else:
        return None, None  # Return None if no emotions are detected

# Apply emotion evaluation and extract emotion and score into separate columns
df[['Emotion', 'Emotion Score']] = df['Transcript'].apply(lambda text: pd.Series(evaluate_emotion(text)))

print(df[['Transcript', 'Emotion', 'Emotion Score']])

df.head()

df.drop(columns = ['Emotions','Score'], inplace =True)

df.head()

def calculate_mispronounced_words(text):  # Removed 'error_count' parameter
    """
    Calculates the number of mispronounced words in a text and returns a score out of 10.

    Args:
        text (str): The text to analyze.

    Returns:
        float: A score out of 10 representing the proportion of correctly pronounced words.
    """
    correctly_spelled_words = [
    "hello", "hi", "good", "morning", "afternoon", "evening",
    "my", "name", "is", "i'm", "am",
    "pleased", "meet", "you",
    "currently", "working", "at", "as", "a", "an",
    "position", "role", "company", "organization",
    "years", "experience", "in", "field", "industry",
    "passionate", "about", "interested",
    "skills", "include", "expertise", "proficient",
    "looking", "forward", "to", "learning", "contributing",
    "team", "collaborate", "communication", "professional",
    "etiquette", "respect", "punctuality", "integrity",
    "thank", "for", "your", "time", "consideration"
        ]



    words = text.split()
    mispronounced_count = 0  # Start with 0 mispronunciations

    for word in words:
        word = word.lower().strip('.,!?')  # Normalize word for comparison
        if word not in correctly_spelled_words:
            mispronounced_count += 1

    total_words = len(words)
    if total_words == 0:
        return 10  # Assume perfect score if no words

    # Calculate mispronunciation score
    mispronounced_score = max(0, 10 - (mispronounced_count / total_words) * 10)
    return mispronounced_score

# Apply the function to the 'Transcript' column in your DataFrame 'df'
df['Mispronunciation Score'] = df['Transcript'].apply(calculate_mispronounced_words)

print(df[['Transcript', 'Mispronunciation Score']])

df.head()

def calculate_speech_rate(audio_file, text):
    duration = librosa.get_duration(filename=audio_file)
    words = text.split()
    num_words = len(words)
    words_per_minute = (num_words / duration) * 60
    min_wpm = 100  # Minimum words per minute
    max_wpm = 160  # Maximum words per minute

    if words_per_minute < min_wpm:
        speech_rate_score = max(0, 10 - ((min_wpm - words_per_minute) / min_wpm) * 10)
    elif words_per_minute > max_wpm:
        speech_rate_score = max(0, 10 - ((words_per_minute - max_wpm) / max_wpm) * 10)
    else:
        speech_rate_score = 10  # Perfect score if within acceptable range

    return speech_rate_score

df['Speech_Rate_Score'] = df['Transcript'].apply(calculate_mispronounced_words)

df.head()

# Apply the lambda function to scale 'Emotion Score'
df['Emotion Score'] = df['Emotion Score'].apply(lambda score: score * 10)

df.head()

def detect_pauses(audio_file, min_silence_duration=1.0, top_db=20):
    y, sr = librosa.load(audio_file)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    pauses = []
    for i in range(1, len(non_silent_intervals)):
        start = non_silent_intervals[i-1][1]
        end = non_silent_intervals[i][0]
        pause_duration = (end - start) / sr
        if pause_duration >= min_silence_duration:
            pauses.append(pause_duration)
    num_pauses = max(0, len(pauses) - 2)
    max_pauses = 10
    if num_pauses > max_pauses:
        pause_score = max(0, 10 - ((num_pauses - max_pauses) / max_pauses) * 10)
    else:
        pause_score = 10

    return pause_score

df['detect_pausein_speech_Score'] = df['Transcript'].apply(calculate_mispronounced_words)

df.head()

# prompt: Using dataframe df: give me the overall score ciolumn by adding all the scores from each cols and scale it to 10

# Create a new column 'Overall Score' by summing relevant scores
df['Overall Score'] = df['Relevance Score'] + df['Grammar Score'] + df['Emotion Score'] + df['Mispronunciation Score'] + df['Speech_Rate_Score'] + df['detect_pausein_speech_Score']

# Scale the 'Overall Score' to a maximum of 10
max_score = df['Overall Score'].max()
df['Overall Score'] = (df['Overall Score'] / max_score) * 10

# Display the DataFrame with the new 'Overall Score' column
df.head()

df.to_excel("/content/Overall_scores.xlsx")