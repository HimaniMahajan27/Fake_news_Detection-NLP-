# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

This project aims to detect fake news articles using two powerful machine learning algorithms:

* **Logistic Regression**
* **XGBoost Classifier**

We use **TF-IDF vectorization** for feature extraction and evaluate the models using **5-fold cross-validation**, along with **test set evaluation**, **classification reports**, and **confusion matrices**.

---

## 📁 Dataset

We use two CSV datasets:

* `Fake.csv` — Contains fake news articles.
* `True.csv` — Contains real news articles.

Each row includes a `title` and `text`, and we assign:

* `0` → Fake News
* `1` → Real News

✅ After loading and labeling, we merge and shuffle the datasets.

---

## 🧹 Text Preprocessing

The following steps are used for preprocessing:

* Lowercasing all text
* Removing URLs, HTML tags, punctuations
* Removing newline characters
* Removing words with digits

We combine the `title` and `text` fields and apply the cleaning function to create a single `text` column.

---

## 🧠 Feature Extraction

We use `TfidfVectorizer` to convert text into numerical feature vectors with the following config:

```python
TfidfVectorizer(stop_words='english', max_df=0.7)
```

---

## 🔁 Model Validation (Cross-Validation)

We apply 5-Fold Cross-Validation to evaluate both models:

* **Logistic Regression**
* **XGBoost**

Example Output:

```bash
📘 Logistic Regression - Mean Accuracy (Validation): 0.9864
📙 XGBoost - Mean Accuracy (Validation): 0.9852
```

---

## 🚂 Model Training & Testing

### Logistic Regression

* Trained on 80% data
* Evaluated on 20% test set
* Metrics: Accuracy, Precision, Recall, F1 Score

### XGBoost Classifier

* Trained and evaluated similarly to Logistic Regression

---

## 📊 Evaluation Metrics

Metrics printed for both models:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### 🔷 Sample Confusion Matrix (Logistic Regression)

![Confusion Matrix](path/to/confusion_matrix.png)

---

## 📉 Model Comparison (Visualization)

Bar chart comparing 5-fold CV accuracy of both models:

```python
plt.bar(models, mean_scores, color=['skyblue', 'orange'])
plt.title("Model Comparison")
plt.ylabel("Validation Accuracy")
```

![Model Comparison](path/to/model_comparison.png)

---

## 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install via pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

```python
python fake_news_detection.py
```

Or run on [Google Colab](https://colab.research.google.com/) by uploading the notebook and datasets.

---

## 💡 Future Work

* Add more models like Random Forest, SVM
* Use pre-trained embeddings (Word2Vec, BERT)
* Implement live prediction UI with Streamlit

---

## 🙌 Acknowledgements

Thanks to Kaggle for the datasets and open-source community for the libraries.
