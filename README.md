AI-Based Fake News & Misinformation Detection System

Project Overview - 

This project is an AI-based system that detects whether a news article is REAL or FAKE using Natural Language Processing (NLP).
It also highlights suspicious words and provides an explanation for the prediction.
The system was built as part of a machine test to demonstrate:
        Text classification
        Explainable AI
        Model deployment with Streamlit

Objective

The system:

    Accepts a news article as input
    Classifies it as REAL or FAKE
    Displays a confidence score
    Highlights important words influencing prediction
    Generates a credibility explanation

Technologies Used
    Python
    Scikit-learn
    TF-IDF Vectorization
    Logistic Regression
    LIME (Explainable AI)
    Streamlit (Web App UI)

Dataset
    Dataset used:
        Fake and Real News Dataset (Kaggle)
    Contains:
        Fake.csv
        True.csv

Each article is labeled as:
    0 → Fake
    1 → Real

Model Training (Done in Google Colab)

Steps:
    Load dataset
    Preprocess text
    Convert text → TF-IDF features
    Train Logistic Regression model

Evaluate using:
    Accuracy
    Precision
    Recall
    F1 Score

    Save model and vectorizer

Saved files:
    fake_news_model.pkl
    vectorizer.pkl

Model Performance
    Example metrics:

        Accuracy: ~0.98
        Precision: ~0.98
        Recall: ~0.98
        F1 Score: ~0.98

Streamlit Application
    The web app allows users to:
    Enter news text
    Get REAL/FAKE prediction
    View confidence score
    See highlighted suspicious words
    Read explanation