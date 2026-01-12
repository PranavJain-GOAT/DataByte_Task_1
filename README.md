🚨 Real vs Fake Disaster Tweets Classification
📌 Project Overview

This project solves the “Real or Fake Disaster Tweets” problem using Natural Language Processing (NLP).
The task is to classify tweets as real disaster-related or not, despite the presence of heavy noise, informal language, and semantic ambiguity.

The project strictly follows the given constraints:

❌ No AutoML

✅ Manual preprocessing

✅ Multiple model approaches

✅ F1-score–centric evaluation

🎯 Problem Statement

Given a tweet, predict whether it refers to a real disaster event (1) or not (0).

Key challenges:

URLs, mentions, emojis, hashtags

Informal & conversational language

Metaphorical use of disaster terms

Class imbalance

🧠 Dataset

Source: Kaggle – NLP Getting Started

Training size: ~7,600 tweets

Target labels:

1 → Real Disaster

0 → Not a Disaster

Imbalance: Moderate → accuracy is misleading

🔍 Exploratory Data Analysis (EDA)

The notebook includes EDA to:

Visualize class distribution

Confirm imbalance

Identify noise patterns such as:

URLs (http, https)

Mentions (@username)

Hashtags

Non-ASCII characters

EDA results directly informed preprocessing and metric choice.

🧹 Text Preprocessing Pipeline

A custom text-cleaning function was applied consistently across all models:

Convert text to lowercase

Remove URLs

Remove HTML tags

Remove mentions and hashtags

Remove punctuation and non-alphabetic characters

Normalize whitespace

This ensured fair model comparison and reduced noise impact.

🧪 Models Implemented
1️⃣ TF-IDF + Logistic Regression (Classic Baseline)

Pipeline

TF-IDF Vectorizer

Logistic Regression classifier

Purpose

Establish a strong and interpretable baseline

Benchmark against deep learning models

Evaluation

Evaluated using F1-score

Confusion matrix analyzed

Performed strongly despite simplicity

2️⃣ LSTM-Based Deep Learning Model

Architecture

Tokenizer + padded sequences

Embedding layer

LSTM layer

Dense sigmoid output

Training Details

Binary Cross-Entropy loss

Adam optimizer

Early stopping to prevent overfitting

Observations

Captures word order and sequential context

Performance sensitive to data size and threshold choice

3️⃣ BERT (Transformer-Based Model)

Model

Pretrained BERT-base

Fine-tuned for binary classification

Why BERT

Bidirectional contextual embeddings

Superior handling of semantic ambiguity

State-of-the-art NLP architecture

Implementation Details

Hugging Face Transformers

Tokenization with attention masks

Fine-tuning on disaster tweet data

Sigmoid output for binary classification

Outcome

Best-performing model in the notebook

Stronger understanding of context-heavy and metaphorical tweets

📊 Evaluation Metric

Primary Metric: F1-Score

Why F1?

Dataset is imbalanced

Balances precision and recall

Penalizes both false positives and false negatives

Accuracy was intentionally not used as the deciding metric.

📈 Model Performance Summary
Model	Performance Summary
TF-IDF + Logistic Regression	Strong baseline
LSTM	Comparable, data-sensitive
BERT (Fine-Tuned)	Best overall performance

📌 Exact scores, classification reports, and confusion matrices are shown in the notebook output cells, ensuring full transparency and reproducibility.

📤 Submission File

submission.csv is generated using the best-performing model (BERT)

Format strictly follows Kaggle submission requirements:

id,target

📁 Project Structure
├── project.ipynb
├── README.md
├── submission.csv
├── train.csv
└── test.csv

⚙️ How to Run
pip install numpy pandas scikit-learn tensorflow torch transformers
jupyter notebook project.ipynb

🏁 Final Conclusion

Classic ML models remain highly competitive

LSTM adds sequence awareness but is data-limited

BERT significantly outperforms other approaches

Proper preprocessing + metric choice matters more than model hype

This project demonstrates:

End-to-end NLP pipeline design

Responsible metric selection

Fair model comparison

Real-world text handling

📌 Possible Improvements

Threshold tuning for F1 maximization

Error analysis on false positives

Lightweight transformers (DistilBERT)

Ensemble approaches

✅ Task Compliance Checklist

✔ No AutoML
✔ Manual preprocessing
✔ Multiple modeling approaches
✔ F1-score focused evaluation
✔ Transformer model implemented
✔ Submission file generated
