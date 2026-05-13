# 📩 Spam Text Classification From Scratch

## 📌 Project Overview
This project focuses on building a robust **Spam-vs-Ham text classifier** entirely from first principles. Developed as part of a Machine Learning course at the **Holon Institute of Technology (HIT)**, the core objective was to implement the mathematical logic of machine learning algorithms without relying on high-level "black-box" libraries like `scikit-learn` for the modeling phase.

The system processes raw SMS text messages and classifies them as either:

- **Ham** → legitimate messages  
- **Spam** → unsolicited or promotional messages  

---

# 🚀 The Core Concept (For Non-Technical Readers)

Imagine trying to teach someone to distinguish between an important letter and a junk mail advertisement.

Instead of giving strict rules, you show them thousands of examples and say:

> “Pay attention to the kinds of words that usually appear in spam messages.”

Over time, they begin noticing patterns:
- Messages containing words like **"FREE"**, **"WINNER"**, or **"PRIZE"** are often spam.
- Messages like **"Hi Mom"** or **"See you tomorrow"** are usually normal.

That is exactly what this project does.

We built a small "digital brain" that learns the probability of certain words appearing in spam messages versus regular messages.

---

# 🧠 How The System Works

## 1. Data Preparation (Preprocessing)

Computers do not understand human language directly — they understand numbers.

To solve this problem, we transformed every SMS message into a numerical representation by:
- Cleaning the text
- Removing duplicates
- Splitting sentences into words
- Counting how often words appear

This process allows the computer to mathematically analyze text.

---

## 2. Building The Model From Scratch

Instead of using ready-made machine learning models, we implemented the mathematical logic ourselves using **NumPy** and **Pandas**.

### 📌 Implemented Model:
### **Multinomial Naive Bayes**

This model works using probabilities.

For example, the system asks questions like:
- “How likely is this message to be spam if it contains the word *Winner*?”
- “How often does the word *Free* appear in spam compared to normal messages?”

The model combines these probabilities and makes a final prediction.

---

## 3. Preventing Model Mistakes

### Laplace Smoothing
Sometimes the model encounters words it has never seen before.

Without protection, this could completely break the probability calculations.

To solve this, we implemented **Laplace Smoothing**, which prevents zero-probability problems and makes the model more stable.

---

# 🔤 Feature Engineering

We implemented a custom vectorizer (`ScratchVectorizer`) supporting multiple text representations:

- **Bag of Words (BoW)**  
  Counts how many times each word appears.

- **TF-IDF**  
  Gives higher importance to unique and informative words while reducing the weight of very common words.

- **Binary Representation**  
  Stores whether a word exists in the message or not.

---

# ⚙️ Training Pipeline

The project includes a full machine learning workflow:

1. Splitting data into:
   - Training set
   - Validation set
   - Test set

2. Training the custom model

3. Testing multiple hyperparameters

4. Choosing the best-performing configuration

5. Evaluating the final model on completely unseen messages

---

# 📊 Evaluation Metrics

To measure performance, we implemented our own evaluation functions for:

- **Accuracy**  
  Overall correctness of the predictions.

- **Precision**  
  Measures how many messages predicted as spam were actually spam.

- **Recall**  
  Measures how much spam the model successfully detected.

- **F1-Score**  
  A balanced metric combining Precision and Recall.

- **Confusion Matrix**  
  A visual representation showing:
  - Correct predictions
  - Missed spam messages
  - False alarms

---

# 📈 Results

The final model achieved excellent performance on unseen test data:

- **Accuracy:** ~98.8%
- **High Spam Precision**
- **High Spam Recall**
- Very low number of false positive predictions

This means the model successfully catches spam messages while rarely marking legitimate messages as spam.

---

# 🛠 Libraries Used

To satisfy the “from scratch” requirement, we used only foundational libraries:

- **NumPy** → numerical computations and probability calculations  
- **Pandas** → data loading and manipulation  
- **Matplotlib & Seaborn** → visualization and confusion matrix plotting  
- **Regex (`re`)** → custom text cleaning and tokenization  

⚠️ No built-in machine learning models such as `LogisticRegression` or `MultinomialNB` were used.

---

# 🌟 Bonus Features

The project also includes:

- Handling imbalanced datasets using:
  - Oversampling
  - Undersampling

- Explainability tools that identify:
  - Which words most strongly influence spam predictions

This helps understand *why* the model made a specific decision.

---

# 👥 Authors

- **Isabelle Ditsev**
- **Hila Tati**
- **Noam Rahcamim**

---

# 🏫 Academic Context

Project developed as part of a Machine Learning course at:

**Holon Institute of Technology (HIT)**

---