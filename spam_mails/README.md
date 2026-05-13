# 📩 Spam Text Classification From Scratch

## 📌 Project Overview
This project implements a **Spam vs Ham SMS classifier from scratch** as part of a Machine Learning course at the **Holon Institute of Technology (HIT)**.

The main goal was to build a complete machine learning pipeline **without using high-level ML libraries (such as scikit-learn for modeling)**, focusing on understanding and implementing the underlying mathematical principles.

The model classifies SMS messages into:
- **Ham** → legitimate messages  
- **Spam** → unsolicited / promotional messages  

---

## 🚀 Project Pipeline

### 1. Data Preprocessing
- Cleaning raw SMS text
- Removing duplicates
- Tokenizing messages using custom logic

### 2. Feature Engineering
- Converting text into numerical representations using a custom vectorizer

### 3. Model Implementation
- Implemented **Multinomial Naive Bayes from scratch**
- No built-in ML classifiers were used

### 4. Model Optimization
- Hyperparameter tuning using validation data

### 5. Handling Class Imbalance
- Oversampling and undersampling techniques to improve performance on imbalanced data

### 6. Explainability
- Analysis of most influential words contributing to spam classification

---

## 🧠 Model Description

### Multinomial Naive Bayes (From Scratch)
A probabilistic classifier based on **Bayes' Theorem**, assuming conditional independence between words.

Key techniques used:
- **Laplace Smoothing** → prevents zero probabilities
- **Log-space computations** → prevents numerical underflow

---

## 🔤 Feature Representation

We implemented a custom text vectorizer (`ScratchVectorizer`) supporting:

- **Bag of Words (Count-based representation)**
- **TF-IDF weighting**
- **Binary representation (0/1 word presence)**

---

## 🛠️ Technologies Used

Only foundational libraries were used:

- **NumPy** → numerical computations and linear algebra  
- **Pandas** → data handling and preprocessing  
- **Regex (re)** → text cleaning and tokenization  
- **Matplotlib / Seaborn** → evaluation visualizations  

---

## 📊 Evaluation Metrics

The model was evaluated using custom implementations of:

- **Accuracy** → overall correctness  
- **Precision** → minimizing false spam predictions  
- **Recall** → detecting as much spam as possible  
- **F1 Score** → balance between precision and recall  

---

## 📈 Results

Performance on the test set:

- **Accuracy:** ~98.8%  
- **Spam Precision:** High (low false positives)  
- **Spam Recall:** High (strong spam detection ability)  

---

## 👥 Team Members

- Noa Rahamim (נועה רחמים)  
- Noa Zadok  
- Liran Sternberg  

---

## 🏫 Academic Context
Developed as part of a Machine Learning course at **Holon Institute of Technology (HIT)**.

---

## 📌 Notes
- This project was built entirely from scratch for educational purposes.
- No pre-built machine learning models were used.