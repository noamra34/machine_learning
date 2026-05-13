import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path("spam_text_classification_from_scratch.ipynb")


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    md_cell(
        """
        # Spam Text Classification From Scratch

        **Student:** נועה  
        **Dataset:** `SPAM text message 20170820 - Data.csv`

        This notebook builds a complete **spam-vs-ham text classifier from scratch**.  
        We will **not** use ready-made machine learning models such as `LogisticRegression` or `MultinomialNB`.

        Instead, we will implement the core pieces ourselves:

        1. quality metrics,
        2. text preprocessing,
        3. a custom vectorizer,
        4. a custom learning algorithm,
        5. hyperparameter search,
        6. imbalance handling and explainability.

        The goal is not only to get good predictions, but also to understand the **mathematical logic** behind each step.
        """
    ),
    md_cell(
        """
        ## Roadmap

        The notebook follows the assignment structure exactly:

        - **Part 1 - Introduction & Quality Metric**
        - **Part 2 - Feature Engineering**
        - **Part 3 - Learning Algorithm Implementation**
        - **Part 4 - Training Flow**
        - **Part 5 - Prediction & Evaluation**
        - **Part 6 - Extensions & Bonus**

        We will use a **Multinomial Naive Bayes** classifier implemented from scratch with `numpy`.

        Why this choice?

        - It is mathematically clean.
        - It works very well on Bag-of-Words text data.
        - Its parameters are interpretable, so it is excellent for explainability.
        """
    ),
    code_cell(
        """
        import math
        import re
        from collections import Counter
        from itertools import product
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        RANDOM_STATE = 42
        DATA_PATH = Path("SPAM text message 20170820 - Data.csv")

        np.random.seed(RANDOM_STATE)
        pd.set_option("display.max_colwidth", 120)
        """
    ),
    md_cell(
        """
        ## Part 1 - Introduction & Quality Metric

        ### 1.1 The learning problem

        We are solving a **binary text classification** task:

        - class `0` = **ham** (normal message),
        - class `1` = **spam** (unsolicited / promotional / scam-like message).

        Each example is a text message.  
        Our model must learn a mapping:

        $$
        f(\\text{message}) \\rightarrow \\{0,1\\}
        $$

        This is a supervised learning problem because each message in the dataset already has a correct label.

        ### 1.2 Why accuracy alone is not enough

        This dataset is imbalanced: there are usually many more ham messages than spam messages.
        Because of that, a model could get high **accuracy** while still missing many spam examples.

        For spam filtering, we care about:

        - **Precision**: when the model says "spam", how often is it correct?
        - **Recall**: out of all real spam messages, how many did we catch?
        - **F1-score**: the harmonic mean of precision and recall.

        ### 1.3 Confusion matrix definitions

        For the positive class = spam:

        - **TP**: predicted spam and actually spam
        - **TN**: predicted ham and actually ham
        - **FP**: predicted spam but actually ham
        - **FN**: predicted ham but actually spam

        The main metrics are:

        $$
        \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}
        $$

        $$
        \\text{Precision} = \\frac{TP}{TP + FP}
        $$

        $$
        \\text{Recall} = \\frac{TP}{TP + FN}
        $$

        $$
        F_1 = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}
        $$

        We will implement these metrics ourselves instead of calling a library helper.
        """
    ),
    code_cell(
        """
        def load_dataset(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            expected_columns = {"Category", "Message"}
            if set(df.columns) != expected_columns:
                raise ValueError(f"Expected columns {expected_columns}, got {set(df.columns)}")
            df = df.copy()
            df["Message"] = df["Message"].astype(str)
            df["Category"] = df["Category"].astype(str).str.strip().str.lower()
            df["label"] = df["Category"].map({"ham": 0, "spam": 1})
            if df["label"].isna().any():
                bad = df.loc[df["label"].isna(), "Category"].unique()
                raise ValueError(f"Unexpected labels found: {bad}")
            df["label"] = df["label"].astype(int)
            return df


        data_raw = load_dataset(DATA_PATH)
        print("Raw dataset shape:", data_raw.shape)
        print()
        print(data_raw["Category"].value_counts())
        display(data_raw.head())
        """
    ),
    code_cell(
        """
        duplicate_rows = data_raw.duplicated().sum()
        missing_values = data_raw[["Category", "Message"]].isna().sum()

        print("Duplicate rows:", duplicate_rows)
        print("Missing values:")
        print(missing_values)

        # Removing exact duplicate rows avoids leaking repeated messages across train/validation/test.
        data = data_raw.drop_duplicates().reset_index(drop=True)
        print()
        print("Shape after duplicate removal:", data.shape)
        print(data["Category"].value_counts())
        """
    ),
    md_cell(
        """
        ### 1.4 Implementing the metrics from scratch

        The following code uses only basic arithmetic and array operations.
        Notice that we explicitly choose **spam** as the positive class because that is the class we most care about detecting.
        """
    ),
    code_cell(
        """
        def confusion_matrix_counts(y_true, y_pred, positive_label=1):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
            tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
            fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
            fn = np.sum((y_true == positive_label) & (y_pred != positive_label))

            return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


        def accuracy_score_manual(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            return float(np.mean(y_true == y_pred))


        def precision_score_manual(y_true, y_pred, positive_label=1):
            counts = confusion_matrix_counts(y_true, y_pred, positive_label)
            denom = counts["tp"] + counts["fp"]
            return counts["tp"] / denom if denom else 0.0


        def recall_score_manual(y_true, y_pred, positive_label=1):
            counts = confusion_matrix_counts(y_true, y_pred, positive_label)
            denom = counts["tp"] + counts["fn"]
            return counts["tp"] / denom if denom else 0.0


        def f1_score_manual(y_true, y_pred, positive_label=1):
            precision = precision_score_manual(y_true, y_pred, positive_label)
            recall = recall_score_manual(y_true, y_pred, positive_label)
            denom = precision + recall
            return 2 * precision * recall / denom if denom else 0.0


        def balanced_accuracy_manual(y_true, y_pred, positive_label=1):
            counts = confusion_matrix_counts(y_true, y_pred, positive_label)
            recall_positive = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
            recall_negative = counts["tn"] / (counts["tn"] + counts["fp"]) if (counts["tn"] + counts["fp"]) else 0.0
            return 0.5 * (recall_positive + recall_negative)


        def metric_summary(y_true, y_pred, positive_label=1):
            counts = confusion_matrix_counts(y_true, y_pred, positive_label)
            return {
                "accuracy": accuracy_score_manual(y_true, y_pred),
                "precision": precision_score_manual(y_true, y_pred, positive_label),
                "recall": recall_score_manual(y_true, y_pred, positive_label),
                "f1": f1_score_manual(y_true, y_pred, positive_label),
                "balanced_accuracy": balanced_accuracy_manual(y_true, y_pred, positive_label),
                **counts,
            }


        # Tiny sanity-check example
        y_true_demo = np.array([1, 1, 1, 0, 0, 0])
        y_pred_demo = np.array([1, 0, 1, 0, 1, 0])
        pd.Series(metric_summary(y_true_demo, y_pred_demo)).round(4)
        """
    ),
    md_cell(
        """
        ## Part 2 - Feature Engineering

        Machine learning models cannot read raw text directly.  
        We first need to convert each message into a numeric vector.

        ### 2.1 Text preprocessing

        We will implement a simple custom pipeline:

        1. convert text to lowercase,
        2. extract tokens with a regular expression,
        3. count how many times each token appears.

        This design is intentionally simple and transparent.  
        In academic assignments, a simple pipeline that you can explain clearly is often better than a complicated one you cannot justify.

        ### 2.2 Bag of Words

        Suppose our vocabulary contains $V$ unique words.
        Then each message becomes a vector:

        $$
        x = [x_1, x_2, \\dots, x_V]
        $$

        where $x_j$ is the count of the $j$-th vocabulary word in that message.

        This is called the **Bag-of-Words** representation because it ignores word order and only keeps word frequencies.

        ### 2.3 Optional TF-IDF representation

        We will also implement TF-IDF to compare feature choices.

        For term $t$ in document $d$:

        $$
        TF(t,d) = \\frac{\\text{count}(t,d)}{\\sum_k \\text{count}(k,d)}
        $$

        $$
        IDF(t) = \\log\\left(\\frac{1 + N}{1 + DF(t)}\\right) + 1
        $$

        $$
        TFIDF(t,d) = TF(t,d) \\cdot IDF(t)
        $$

        where:

        - $N$ is the number of training documents,
        - $DF(t)$ is the number of documents containing term $t$.

        TF-IDF reduces the weight of very common words and increases the influence of more distinctive words.
        """
    ),
    code_cell(
        """
        class ScratchVectorizer:
            def __init__(self, min_df=1, max_features=None, representation="count", lowercase=True):
                if representation not in {"count", "binary", "tfidf"}:
                    raise ValueError("representation must be 'count', 'binary', or 'tfidf'")
                self.min_df = min_df
                self.max_features = max_features
                self.representation = representation
                self.lowercase = lowercase

                self.vocabulary_ = None
                self.feature_names_ = None
                self.document_frequency_ = None
                self.idf_ = None
                self.n_documents_ = None

            def _tokenize(self, text):
                text = str(text)
                if self.lowercase:
                    text = text.lower()
                return re.findall(r"[a-z0-9']+", text)

            def fit(self, documents):
                term_counts = Counter()
                document_frequency = Counter()

                for doc in documents:
                    tokens = self._tokenize(doc)
                    term_counts.update(tokens)
                    document_frequency.update(set(tokens))

                candidate_terms = [
                    term for term in term_counts
                    if document_frequency[term] >= self.min_df
                ]

                candidate_terms = sorted(candidate_terms, key=lambda term: (-term_counts[term], term))
                if self.max_features is not None:
                    candidate_terms = candidate_terms[: self.max_features]

                self.feature_names_ = candidate_terms
                self.vocabulary_ = {term: idx for idx, term in enumerate(candidate_terms)}
                self.document_frequency_ = np.array(
                    [document_frequency[term] for term in candidate_terms],
                    dtype=np.float64,
                )
                self.n_documents_ = len(list(documents)) if not isinstance(documents, list) else len(documents)

                if self.representation == "tfidf":
                    self.idf_ = np.log((1 + self.n_documents_) / (1 + self.document_frequency_)) + 1.0
                else:
                    self.idf_ = None

                return self

            def transform(self, documents):
                if self.vocabulary_ is None:
                    raise ValueError("Vectorizer must be fit before calling transform.")

                n_docs = len(documents)
                n_features = len(self.vocabulary_)
                X = np.zeros((n_docs, n_features), dtype=np.float32)

                for row_idx, doc in enumerate(documents):
                    token_counts = Counter(self._tokenize(doc))
                    for token, count in token_counts.items():
                        col_idx = self.vocabulary_.get(token)
                        if col_idx is not None:
                            X[row_idx, col_idx] = count

                if self.representation == "binary":
                    X = (X > 0).astype(np.float32)

                elif self.representation == "tfidf":
                    row_sums = X.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    term_frequency = X / row_sums
                    X = term_frequency * self.idf_.astype(np.float32)

                return X

            def fit_transform(self, documents):
                documents = list(documents)
                self.fit(documents)
                return self.transform(documents)

            def get_feature_names_out(self):
                return np.array(self.feature_names_)
        """
    ),
    code_cell(
        """
        sample_vectorizer = ScratchVectorizer(min_df=1, max_features=20, representation="count")
        sample_X = sample_vectorizer.fit_transform(data["Message"].head(5).tolist())

        print("Sample vocabulary:")
        print(sample_vectorizer.get_feature_names_out())
        print()
        print("Feature matrix shape:", sample_X.shape)
        pd.DataFrame(sample_X, columns=sample_vectorizer.get_feature_names_out()).head()
        """
    ),
    md_cell(
        """
        ### Why we fit the vectorizer only on training data

        This is an important machine learning rule.

        If we build the vocabulary using the full dataset before splitting, then information from the validation or test set leaks into training.
        Even though vocabulary creation looks innocent, it still uses future data.

        So our rule is:

        - **fit** the vectorizer on the training set only,
        - **transform** validation/test using the already learned vocabulary.
        """
    ),
    md_cell(
        """
        ## Part 3 - Learning Algorithm Implementation

        We now implement **Multinomial Naive Bayes** from scratch.

        ### 3.1 Intuition

        We want to compute:

        $$
        P(y \\mid x)
        $$

        where:

        - $y \\in \\{\\text{ham}, \\text{spam}\\}$ is the class,
        - $x$ is the document feature vector.

        Bayes' rule gives:

        $$
        P(y \\mid x) = \\frac{P(x \\mid y)P(y)}{P(x)}
        $$

        Since $P(x)$ is the same for every class during comparison, we only need:

        $$
        P(x \\mid y)P(y)
        $$

        ### 3.2 Naive assumption

        The "naive" assumption says that, given the class, word occurrences are conditionally independent.
        This is not literally true in natural language, but it often works surprisingly well.

        For the multinomial model:

        $$
        P(x \\mid y) \\propto \\prod_{j=1}^{V} P(w_j \\mid y)^{x_j}
        $$

        Taking logs turns multiplication into addition:

        $$
        \\log P(y \\mid x) \\propto \\log P(y) + \\sum_{j=1}^{V} x_j \\log P(w_j \\mid y)
        $$

        This is why Naive Bayes is so efficient for text.

        ### 3.3 Laplace smoothing

        Without smoothing, if a word never appeared in spam during training, then:

        $$
        P(w_j \\mid \\text{spam}) = 0
        $$

        and the whole product becomes zero. That is too brittle.

        So we use **Laplace smoothing**:

        $$
        P(w_j \\mid y) =
        \\frac{N_{y,j} + \\alpha}{\\sum_{k=1}^{V} N_{y,k} + \\alpha V}
        $$

        where:

        - $N_{y,j}$ = total count of word $j$ inside class $y$,
        - $\\alpha > 0$ is the smoothing hyperparameter.
        """
    ),
    code_cell(
        """
        class ScratchMultinomialNB:
            def __init__(self, alpha=1.0):
                self.alpha = alpha
                self.classes_ = None
                self.class_log_prior_ = None
                self.feature_log_prob_ = None
                self.feature_count_ = None
                self.class_count_ = None

            def fit(self, X, y):
                X = np.asarray(X, dtype=np.float64)
                y = np.asarray(y)

                self.classes_ = np.array(sorted(np.unique(y)))
                n_classes = len(self.classes_)
                n_features = X.shape[1]

                self.class_count_ = np.zeros(n_classes, dtype=np.float64)
                self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

                for class_idx, class_label in enumerate(self.classes_):
                    X_class = X[y == class_label]
                    self.class_count_[class_idx] = X_class.shape[0]
                    self.feature_count_[class_idx] = X_class.sum(axis=0)

                self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

                smoothed_feature_count = self.feature_count_ + self.alpha
                smoothed_totals = smoothed_feature_count.sum(axis=1, keepdims=True)
                self.feature_log_prob_ = np.log(smoothed_feature_count / smoothed_totals)

                return self

            def predict_log_proba(self, X):
                X = np.asarray(X, dtype=np.float64)
                return X @ self.feature_log_prob_.T + self.class_log_prior_

            def predict_proba(self, X):
                log_scores = self.predict_log_proba(X)
                max_log = np.max(log_scores, axis=1, keepdims=True)
                stabilized = np.exp(log_scores - max_log)
                return stabilized / stabilized.sum(axis=1, keepdims=True)

            def predict(self, X):
                log_scores = self.predict_log_proba(X)
                best_class_indices = np.argmax(log_scores, axis=1)
                return self.classes_[best_class_indices]

            def log_odds_ratio(self, positive_label=1):
                if len(self.classes_) != 2:
                    raise ValueError("log_odds_ratio is implemented only for binary classification.")
                pos_idx = int(np.where(self.classes_ == positive_label)[0][0])
                neg_idx = 1 - pos_idx
                return self.feature_log_prob_[pos_idx] - self.feature_log_prob_[neg_idx]
        """
    ),
    code_cell(
        """
        def stratified_train_val_test_split(df, label_col="label", train_size=0.64, val_size=0.16, test_size=0.20, random_state=42):
            if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("train_size + val_size + test_size must equal 1.")

            rng = np.random.default_rng(random_state)
            train_parts, val_parts, test_parts = [], [], []

            for label_value, group in df.groupby(label_col):
                indices = group.index.to_numpy().copy()
                rng.shuffle(indices)

                n = len(indices)
                n_train = int(n * train_size)
                n_val = int(n * val_size)

                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train + n_val]
                test_idx = indices[n_train + n_val:]

                train_parts.append(df.loc[train_idx])
                val_parts.append(df.loc[val_idx])
                test_parts.append(df.loc[test_idx])

            train_df = pd.concat(train_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            val_df = pd.concat(val_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            test_df = pd.concat(test_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            return train_df, val_df, test_df


        train_df, val_df, test_df = stratified_train_val_test_split(data, random_state=RANDOM_STATE)

        print("Train shape:", train_df.shape)
        print("Validation shape:", val_df.shape)
        print("Test shape:", test_df.shape)
        print()

        split_summary = pd.DataFrame({
            "train": train_df["Category"].value_counts(normalize=True).sort_index(),
            "validation": val_df["Category"].value_counts(normalize=True).sort_index(),
            "test": test_df["Category"].value_counts(normalize=True).sort_index(),
        }).round(4)
        display(split_summary)
        """
    ),
    md_cell(
        """
        The class proportions in the three splits are very similar, which is exactly what we want from a **stratified split**.
        """
    ),
    md_cell(
        """
        ## Part 4 - Training Flow

        We now create a small training pipeline that searches over hyperparameters.

        ### 4.1 What are we tuning?

        We will tune:

        - `representation`: `count`, `binary`, or `tfidf`
        - `min_df`: minimum document frequency required for a word to enter the vocabulary
        - `max_features`: vocabulary size cap
        - `alpha`: Laplace smoothing strength

        ### 4.2 Why use validation data?

        The validation set lets us compare multiple candidate models without touching the test set.
        The test set should remain unseen until the very end so that it acts as an honest estimate of generalization.

        ### 4.3 Optimization target

        Because spam detection is imbalanced, we will choose the best model using the **validation F1-score** for the spam class.
        """
    ),
    code_cell(
        """
        def evaluate_configuration(train_messages, train_labels, val_messages, val_labels, vectorizer_params, alpha):
            vectorizer = ScratchVectorizer(**vectorizer_params)
            X_train = vectorizer.fit_transform(train_messages)
            X_val = vectorizer.transform(val_messages)

            model = ScratchMultinomialNB(alpha=alpha)
            model.fit(X_train, train_labels)

            val_pred = model.predict(X_val)
            metrics = metric_summary(val_labels, val_pred, positive_label=1)

            result = {
                **vectorizer_params,
                "alpha": alpha,
                "vocab_size": len(vectorizer.feature_names_),
                **metrics,
            }
            return result


        search_space = {
            "representation": ["count", "binary", "tfidf"],
            "min_df": [1, 2],
            "max_features": [1500, 2500, 4000],
            "alpha": [0.25, 0.5, 1.0, 2.0],
        }

        results = []
        train_messages = train_df["Message"].tolist()
        train_labels = train_df["label"].to_numpy()
        val_messages = val_df["Message"].tolist()
        val_labels = val_df["label"].to_numpy()

        for representation, min_df, max_features, alpha in product(
            search_space["representation"],
            search_space["min_df"],
            search_space["max_features"],
            search_space["alpha"],
        ):
            vectorizer_params = {
                "representation": representation,
                "min_df": min_df,
                "max_features": max_features,
            }
            results.append(
                evaluate_configuration(
                    train_messages=train_messages,
                    train_labels=train_labels,
                    val_messages=val_messages,
                    val_labels=val_labels,
                    vectorizer_params=vectorizer_params,
                    alpha=alpha,
                )
            )

        results_df = pd.DataFrame(results).sort_values(
            by=["f1", "recall", "precision", "accuracy"],
            ascending=False
        ).reset_index(drop=True)

        display(results_df.head(10).round(4))
        """
    ),
    code_cell(
        """
        best_row = results_df.iloc[0].to_dict()
        best_vectorizer_params = {
            "representation": best_row["representation"],
            "min_df": int(best_row["min_df"]),
            "max_features": int(best_row["max_features"]),
        }
        best_alpha = float(best_row["alpha"])

        print("Best validation configuration:")
        print(best_vectorizer_params)
        print("alpha =", best_alpha)
        print()
        print(pd.Series(best_row)[["accuracy", "precision", "recall", "f1", "balanced_accuracy", "vocab_size"]].round(4))
        """
    ),
    code_cell(
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        top_plot = results_df.head(12).copy()
        ax.bar(np.arange(len(top_plot)), top_plot["f1"], color="#1f77b4")
        ax.set_title("Top Validation Configurations by F1")
        ax.set_xlabel("Configuration Rank")
        ax.set_ylabel("Validation F1")
        ax.set_xticks(np.arange(len(top_plot)))
        ax.set_xticklabels(np.arange(1, len(top_plot) + 1))
        plt.show()
        """
    ),
    md_cell(
        """
        ### Interpretation of the search

        A useful pattern to watch for is whether the best model prefers:

        - a larger vocabulary or a smaller one,
        - raw counts or TF-IDF,
        - stronger or weaker smoothing.

        This helps us reason about the dataset instead of treating hyperparameter tuning as a black box.
        """
    ),
    md_cell(
        """
        ## Part 5 - Prediction & Evaluation

        Once the best hyperparameters are selected, we retrain the model on **train + validation** together.
        This gives the final model more labeled data before evaluating on the untouched test set.
        """
    ),
    code_cell(
        """
        train_val_df = pd.concat([train_df, val_df], ignore_index=True)

        final_vectorizer = ScratchVectorizer(**best_vectorizer_params)
        X_train_val = final_vectorizer.fit_transform(train_val_df["Message"].tolist())
        y_train_val = train_val_df["label"].to_numpy()

        X_test = final_vectorizer.transform(test_df["Message"].tolist())
        y_test = test_df["label"].to_numpy()

        final_model = ScratchMultinomialNB(alpha=best_alpha)
        final_model.fit(X_train_val, y_train_val)

        test_pred = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)
        test_metrics = metric_summary(y_test, test_pred, positive_label=1)

        test_metrics_df = pd.DataFrame([test_metrics]).round(4)
        display(test_metrics_df)
        """
    ),
    code_cell(
        """
        def confusion_matrix_dataframe(y_true, y_pred):
            counts = confusion_matrix_counts(y_true, y_pred, positive_label=1)
            return pd.DataFrame(
                [
                    [counts["tn"], counts["fp"]],
                    [counts["fn"], counts["tp"]],
                ],
                index=["Actual ham", "Actual spam"],
                columns=["Predicted ham", "Predicted spam"],
            )


        cm_df = confusion_matrix_dataframe(y_test, test_pred)
        display(cm_df)
        """
    ),
    code_cell(
        """
        fig, ax = plt.subplots(figsize=(5, 4))
        cm_values = cm_df.to_numpy()
        im = ax.imshow(cm_values, cmap="Blues")

        for i in range(cm_values.shape[0]):
            for j in range(cm_values.shape[1]):
                ax.text(j, i, int(cm_values[i, j]), ha="center", va="center", color="black", fontsize=12)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(cm_df.columns)
        ax.set_yticklabels(cm_df.index)
        ax.set_title("Confusion Matrix on the Test Set")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        preview = test_df[["Category", "Message"]].copy()
        preview["predicted_label"] = np.where(test_pred == 1, "spam", "ham")
        preview["spam_probability"] = test_proba[:, list(final_model.classes_).index(1)]

        print("Random test-set examples with predictions:")
        display(preview.sample(10, random_state=RANDOM_STATE).sort_values("spam_probability", ascending=False))
        """
    ),
    md_cell(
        """
        ### Reading the final metrics

        When you discuss results in your assignment, do not only report numbers.
        Explain what they mean:

        - High **precision** means the filter rarely accuses a normal message of being spam.
        - High **recall** means the filter catches most spam messages.
        - High **F1** means the model balances both goals well.

        In many spam settings, missing a spam message (**false negative**) is undesirable, but falsely blocking a legitimate message (**false positive**) is also costly.
        So the right tradeoff depends on the application.
        """
    ),
    md_cell(
        """
        ## Part 6 - Extensions & Bonus

        We now add two more advanced ingredients:

        1. **Imbalance handling** with custom random over-sampling and under-sampling.
        2. **Explainability** by identifying words that strongly push predictions toward spam.
        """
    ),
    code_cell(
        """
        def random_oversample(X, y, random_state=42):
            rng = np.random.default_rng(random_state)
            y = np.asarray(y)
            class_labels, class_counts = np.unique(y, return_counts=True)
            target_count = class_counts.max()

            sampled_indices = []
            for label in class_labels:
                indices = np.where(y == label)[0]
                extra_needed = target_count - len(indices)
                if extra_needed > 0:
                    extra = rng.choice(indices, size=extra_needed, replace=True)
                    indices = np.concatenate([indices, extra])
                sampled_indices.append(indices)

            sampled_indices = np.concatenate(sampled_indices)
            rng.shuffle(sampled_indices)
            return X[sampled_indices], y[sampled_indices]


        def random_undersample(X, y, random_state=42):
            rng = np.random.default_rng(random_state)
            y = np.asarray(y)
            class_labels, class_counts = np.unique(y, return_counts=True)
            target_count = class_counts.min()

            sampled_indices = []
            for label in class_labels:
                indices = np.where(y == label)[0]
                chosen = rng.choice(indices, size=target_count, replace=False)
                sampled_indices.append(chosen)

            sampled_indices = np.concatenate(sampled_indices)
            rng.shuffle(sampled_indices)
            return X[sampled_indices], y[sampled_indices]


        def evaluate_resampling_strategy(strategy_name, X_train, y_train, X_val, y_val, alpha, random_state=42):
            if strategy_name == "none":
                X_used, y_used = X_train, y_train
            elif strategy_name == "oversample":
                X_used, y_used = random_oversample(X_train, y_train, random_state=random_state)
            elif strategy_name == "undersample":
                X_used, y_used = random_undersample(X_train, y_train, random_state=random_state)
            else:
                raise ValueError("Unknown strategy.")

            model = ScratchMultinomialNB(alpha=alpha)
            model.fit(X_used, y_used)
            pred = model.predict(X_val)
            metrics = metric_summary(y_val, pred, positive_label=1)
            return {"strategy": strategy_name, **metrics}
        """
    ),
    code_cell(
        """
        extension_vectorizer = ScratchVectorizer(**best_vectorizer_params)
        X_train_ext = extension_vectorizer.fit_transform(train_df["Message"].tolist())
        y_train_ext = train_df["label"].to_numpy()
        X_val_ext = extension_vectorizer.transform(val_df["Message"].tolist())
        y_val_ext = val_df["label"].to_numpy()

        resampling_results = []
        for strategy in ["none", "oversample", "undersample"]:
            resampling_results.append(
                evaluate_resampling_strategy(
                    strategy_name=strategy,
                    X_train=X_train_ext,
                    y_train=y_train_ext,
                    X_val=X_val_ext,
                    y_val=y_val_ext,
                    alpha=best_alpha,
                    random_state=RANDOM_STATE,
                )
            )

        resampling_df = pd.DataFrame(resampling_results).sort_values(by="f1", ascending=False).reset_index(drop=True)
        display(resampling_df.round(4))
        """
    ),
    code_cell(
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.22
        x = np.arange(len(resampling_df))

        ax.bar(x - width, resampling_df["precision"], width=width, label="Precision")
        ax.bar(x, resampling_df["recall"], width=width, label="Recall")
        ax.bar(x + width, resampling_df["f1"], width=width, label="F1")

        ax.set_xticks(x)
        ax.set_xticklabels(resampling_df["strategy"])
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics Under Different Imbalance Strategies")
        ax.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code_cell(
        """
        best_strategy = resampling_df.iloc[0]["strategy"]
        print("Best resampling strategy on validation:", best_strategy)

        X_train_val_ext = extension_vectorizer.fit_transform(train_val_df["Message"].tolist())
        y_train_val_ext = train_val_df["label"].to_numpy()
        X_test_ext = extension_vectorizer.transform(test_df["Message"].tolist())

        if best_strategy == "oversample":
            X_balanced, y_balanced = random_oversample(X_train_val_ext, y_train_val_ext, random_state=RANDOM_STATE)
        elif best_strategy == "undersample":
            X_balanced, y_balanced = random_undersample(X_train_val_ext, y_train_val_ext, random_state=RANDOM_STATE)
        else:
            X_balanced, y_balanced = X_train_val_ext, y_train_val_ext

        balanced_model = ScratchMultinomialNB(alpha=best_alpha)
        balanced_model.fit(X_balanced, y_balanced)
        balanced_pred = balanced_model.predict(X_test_ext)
        balanced_metrics = metric_summary(y_test, balanced_pred, positive_label=1)

        comparison_df = pd.DataFrame(
            [
                {"model": "baseline", **test_metrics},
                {"model": f"resampled_{best_strategy}", **balanced_metrics},
            ]
        ).round(4)
        display(comparison_df)
        """
    ),
    md_cell(
        """
        ### Explainability: which words push the model toward spam?

        In Multinomial Naive Bayes, each token has a class-conditional probability.
        A useful measure is the **log-odds ratio**:

        $$
        \\log P(w \\mid \\text{spam}) - \\log P(w \\mid \\text{ham})
        $$

        - a large positive value means the token is strongly associated with spam,
        - a large negative value means the token is strongly associated with ham.

        This gives us a transparent explanation for what the model has learned.
        """
    ),
    code_cell(
        """
        def top_influential_tokens(model, vectorizer, top_n=20):
            log_odds = model.log_odds_ratio(positive_label=1)
            feature_names = np.array(vectorizer.feature_names_)

            spam_order = np.argsort(log_odds)[::-1][:top_n]
            ham_order = np.argsort(log_odds)[:top_n]

            spam_df = pd.DataFrame({
                "token": feature_names[spam_order],
                "log_odds_for_spam": log_odds[spam_order],
            })

            ham_df = pd.DataFrame({
                "token": feature_names[ham_order],
                "log_odds_for_spam": log_odds[ham_order],
            })

            return spam_df, ham_df


        explanation_model = balanced_model
        explanation_vectorizer = extension_vectorizer

        top_spam_words, top_ham_words = top_influential_tokens(explanation_model, explanation_vectorizer, top_n=15)

        print("Top words associated with SPAM:")
        display(top_spam_words.round(4))
        print()
        print("Top words associated with HAM:")
        display(top_ham_words.round(4))
        """
    ),
    code_cell(
        """
        def explain_message(model, vectorizer, message, top_n=10):
            X_msg = vectorizer.transform([message])
            log_odds = model.log_odds_ratio(positive_label=1)
            contributions = X_msg[0] * log_odds

            nonzero = np.where(X_msg[0] > 0)[0]
            contribution_df = pd.DataFrame({
                "token": np.array(vectorizer.feature_names_)[nonzero],
                "feature_value": X_msg[0][nonzero],
                "spam_log_odds_contribution": contributions[nonzero],
            }).sort_values("spam_log_odds_contribution", ascending=False)

            prior_diff = model.class_log_prior_[list(model.classes_).index(1)] - model.class_log_prior_[list(model.classes_).index(0)]
            total_log_odds = prior_diff + contributions.sum()
            spam_probability = 1 / (1 + np.exp(-total_log_odds))

            return contribution_df.head(top_n), total_log_odds, spam_probability


        likely_spam_examples = preview[preview["predicted_label"] == "spam"].sort_values("spam_probability", ascending=False)
        example_message = likely_spam_examples.iloc[0]["Message"] if not likely_spam_examples.empty else preview.iloc[0]["Message"]

        explanation_df, total_log_odds, spam_probability = explain_message(
            explanation_model,
            explanation_vectorizer,
            example_message,
            top_n=10,
        )

        print("Example message:")
        print(example_message)
        print()
        print(f"Total spam log-odds: {total_log_odds:.4f}")
        print(f"Estimated spam probability: {spam_probability:.4f}")
        print()
        display(explanation_df.round(4))
        """
    ),
    md_cell(
        """
        ## Final Reflection

        This notebook demonstrates a full machine learning workflow built from first principles:

        - we defined evaluation metrics ourselves,
        - built a vectorizer ourselves,
        - implemented Multinomial Naive Bayes ourselves,
        - tuned hyperparameters ourselves,
        - handled class imbalance ourselves,
        - and interpreted the model ourselves.

        If you are writing this up for your assignment, a good concluding argument is:

        > Text classification works well even with relatively simple models when the representation is sensible.  
        > The combination of Bag-of-Words style features and Naive Bayes is fast, interpretable, and effective for spam detection.

        You can also mention an important limitation:

        - Bag-of-Words ignores word order and context.
        - Naive Bayes assumes conditional independence between words.

        Even with those simplifications, the model can still perform strongly on short text messages.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH.resolve()}")
