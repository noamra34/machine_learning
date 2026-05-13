📩 Spam Text Classification From Scratch
📌 Overview

This project implements a Spam vs Ham text classifier from scratch, as part of a Machine Learning course at the Holon Institute of Technology (HIT).

The goal was to build a complete machine learning pipeline without relying on high-level libraries (e.g., scikit-learn for modeling), focusing instead on implementing the underlying mathematical concepts manually.

The system classifies SMS messages as:

Ham → legitimate messages
Spam → unsolicited / promotional messages
🚀 Pipeline Summary

We built an end-to-end ML workflow including:

Data Preprocessing
Cleaning raw text
Removing duplicates
Tokenization using custom logic
Feature Engineering
Custom vectorization of text into numerical representations
Model Implementation
Multinomial Naive Bayes implemented from scratch
Model Selection
Hyperparameter tuning using validation data
Class Imbalance Handling
Oversampling / undersampling strategies
Explainability
Feature importance analysis (most influential words per class)
🧠 Model
🔹 Multinomial Naive Bayes (From Scratch)

We implemented a probabilistic classifier based on Bayes’ Theorem, assuming conditional independence between words.

Key improvements:

Laplace Smoothing → prevents zero probabilities
Log-space computation → avoids numerical underflow
🔤 Feature Representation
Custom Vectorizer (ScratchVectorizer)

We implemented multiple text representations:

Bag of Words (Count Vectorization)
TF-IDF weighting
Binary representation (0/1 for word presence)
🛠️ Technologies Used

Only foundational libraries were used:

NumPy → linear algebra & numerical computation
Pandas → data handling and preprocessing
Matplotlib / Seaborn → evaluation visualizations
re (Regex) → custom tokenization
📊 Evaluation Metrics

We implemented a full evaluation suite:

Accuracy → overall correctness
Precision → avoiding false spam classification
Recall → capturing as much spam as possible
F1-score → balance between precision and recall
📈 Results

Performance on the test set:

Accuracy: ~98.8%
Spam Precision: High (low false positives)
Spam Recall: High (strong spam detection ability)
👥 Team
Noa Rahamim (נועה רחמים)
Noa Zadok
Liran Sternberg
🏫 Academic Context

Developed at Holon Institute of Technology (HIT) as part of a Machine Learning course project.