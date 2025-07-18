# -*- coding: utf-8 -*-
"""Fake_News_detect.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_rGFBxvJFV_wL0jnPJadeIchvvu3Akdu
"""

!pip install xgboost

# Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

# NLP + ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Step 2: Load Dataset
fake_df = pd.read_csv("/content/Fake.csv")
true_df = pd.read_csv("/content/True.csv")

# Add labels
fake_df["label"] = 0  # Fake
true_df["label"] = 1  # Real

# Combine and shuffle
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# Step 3: Text Preprocessing (NLP)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["text"] = df["title"] + " " + df["text"]
df["text"] = df["text"].apply(clean_text)

# Step 4: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Step 5: Cross-validation (Validation Accuracy)
log_model = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
print("📘 Logistic Regression - Fold Accuracies:", log_scores)
print("📘 Logistic Regression - Mean Accuracy (Validation):", np.mean(log_scores))

xgb_model = XGBClassifier(eval_metric='logloss')
xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')
print("📙 XGBoost - Fold Accuracies:", xgb_scores)
print("📙 XGBoost - Mean Accuracy (Validation):", np.mean(xgb_scores))

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training & Evaluation

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📘 Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred))
print("\n📘 Logistic Regression Report:\n", classification_report(y_test, y_pred))

# Optional: Precision, Recall, F1
print("🔹 Logistic Regression Precision:", precision_score(y_test, y_pred))
print("🔹 Logistic Regression Recall:", recall_score(y_test, y_pred))
print("🔹 Logistic Regression F1 Score:", f1_score(y_test, y_pred))

# XGBoost
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("\n📙 XGBoost Test Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\n📙 XGBoost Report:\n", classification_report(y_test, y_pred_xgb))

# Optional: Precision, Recall, F1
print("🟠 XGBoost Precision:", precision_score(y_test, y_pred_xgb))
print("🟠 XGBoost Recall:", recall_score(y_test, y_pred_xgb))
print("🟠 XGBoost F1 Score:", f1_score(y_test, y_pred_xgb))

# Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# Step 8: Model Comparison Graph
models = ['Logistic Regression', 'XGBoost']
mean_scores = [np.mean(log_scores), np.mean(xgb_scores)]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, mean_scores, color=['skyblue', 'orange'], width=0.5)
plt.ylabel("Validation Accuracy (5-Fold CV)")
plt.title("Model Comparison")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.002, f"{height:.4f}", ha='center', fontsize=12, fontweight='bold')
plt.ylim(0.95, 1.0)
plt.show()