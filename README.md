# Suicide Detection Using Machine Learning

This project implements a comprehensive machine learning pipeline for detecting suicidal ideation in Reddit posts. It uses both classical machine learning models and deep learning approaches, leveraging multiple text embedding techniques and exploring demographic features like gender and age.

---

## Google Colab link
[Google Colab link](https://colab.research.google.com/drive/1HJWPvm4jK0BOjNFtDAbwSCFbR6iKqrbh#scrollTo=pfoxDzDocIHU)

---

## PDF
[PDF Link](https://github.com/IvanGorbache/Machine-Learning-Project/blob/main/%D7%A4%D7%A8%D7%95%D7%99%D7%A7%D7%98%20%D7%9C%D7%9E%D7%99%D7%93%D7%AA%20%D7%9E%D7%9B%D7%95%D7%A0%D7%94.pdf)

---
## Project Overview

The main goals of this project are:

- Detect posts indicative of suicidal ideation using text classification.
- Compare different models and embedding techniques.
- Analyze gender and age distributions and their effect on predictions.
- Use clustering and visualization to better understand the data.
- Perform error analysis to inspect misclassifications and improve model understanding.

---

## Features and Components

### Preprocessing
- Text cleaning (removal of URLs, subreddit/user mentions, and unnecessary symbols).
- Special handling of the word "filler" (an artificial label from the dataset).
- Age and gender extraction via rule-based heuristics.

### Embeddings
- TF-IDF (word frequency-based)
- USE (Universal Sentence Encoder) — semantic sentence embeddings via TensorFlow Hub

### Models
- SVM (LinearSVC + Calibration)
- Logistic Regression
- Random Forest
- Fully Connected Neural Network (TensorFlow/Keras)

### Evaluation
- Metrics at thresholds 0.5 and 0.3 (for higher recall using F2 score)
- Confusion matrices
- Precision, recall, F2 score, classification report

### Demographic Feature Analysis
- Gender and age extraction from text
- SVM model re-evaluation with gender as additional input

### Visualizations
- Word Clouds (for suicide vs. non-suicide)
- Gender and Age group distributions
- Confusion matrices
- Clustering (KMeans + t-SNE)

---

## Clustering

Unsupervised KMeans clustering combined with:
- Dimensionality reduction (SVD + t-SNE)
- Cluster purity analysis
- Side-by-side visualization: clusters vs true labels

---

## Dataset
[Dataset link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

---
