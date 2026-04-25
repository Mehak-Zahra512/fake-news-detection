# Fake News Detection
Classifying news articles as real or fake using NLP —
built to understand how text data can expose misinformation patterns.

## The Problem
Misinformation spreads faster than corrections. I wanted to understand
whether a simple model could catch it — and where it would fail.

## What I Built
A binary text classifier trained on labeled news articles (Fake=0, Real=1)
using TF-IDF vectorisation and Multinomial Naïve Bayes on an 80/20 split.

## Result
**80–85% accuracy** on unseen test data

![Model Output] (output.png)

## Tech Stack
Python · Scikit-learn · Pandas · TF-IDF · Multinomial Naïve Bayes

## Key Learning
Preprocessing choices shifted accuracy more than switching the model.
Stop-word removal via TF-IDF had more impact than I expected —
that lesson changed how I think about feature engineering.

## How to Run
pip install -r requirements.txt
python fake_news_detection.py

## Honest Limitation
This model struggles with paraphrased misinformation —
it relies on vocabulary patterns, not meaning.
Next version: word embeddings or a transformer-based approach.
