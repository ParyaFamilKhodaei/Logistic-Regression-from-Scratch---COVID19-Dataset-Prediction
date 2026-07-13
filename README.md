A project done with my friend: github.com/AnoshaSameti
# COVID-19 Mortality Prediction: Logistic Regression from Scratch 🦠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-orange)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-brightgreen)](https://pandas.pydata.org/)

This repository contains a comprehensive implementation of **Logistic Regression built from the ground up** to predict COVID-19 patient outcomes. The project specifically addresses the challenges of **Class Imbalance** and **Overfitting** in medical datasets.

---

## 🎯 Project Overview
This project was developed as part of the *Fundamentals of Machine Learning & Neural Networks* course. The objective is to predict the probability of survival for COVID-19 patients using clinical features like comorbidities, age, and hospitalization status.

### Key Highlights:
- **Zero-Dependency Learning**: No high-level ML libraries (like Scikit-Learn) were used for model training.
- **Mathematical Foundation**: Implementation of Gradient Descent, Sigmoid activation, and Binary Cross-Entropy.
- **Imbalance Analysis**: A deep dive into how skewed data distributions affect model reliability.
- **Regularization**: Implementation of **L2 (Ridge) Regularization** to manage the Bias-Variance tradeoff.

---

## 🛠 Technical Implementation

### 1. Data Preprocessing Pipeline
The raw clinical data required intensive cleaning to be suitable for an AI model:
- **Missing Value Handling**: Systematic removal of null values and placeholder characters (e.g., `#`).
- **Feature Engineering**: Handling specific encoded values (like `9999-99-99`) in the `date_died` column to create a binary target (1 for deceased, 0 for survived).
- **Normalization**: Manual implementation of Min-Max Scaling and Standardization to ensure stable gradient updates.

### 2. Mathematics of the Model
The model optimizes the following cost function with L2 Regularization:

$$ J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 $$

Where the hypothesis is the Sigmoid function:
$$ f_{w,b}(x) = \frac{1}{1 + e^{-(wx + b)}} $$

### 3. Manual Evaluation Metrics
To strictly follow the "from-scratch" requirement, all performance metrics were calculated manually using the Confusion Matrix:
- **Accuracy**: $(TP + TN) / Total$
- **Precision**: $TP / (TP + FP)$
- **Recall**: $TP / (TP + FN)$
- **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

---

## 📊 Comparative Analysis: Imbalanced vs. Balanced Data

The project explores the **Accuracy Paradox**. In medical diagnostics, a model can have 95% accuracy but 0% recall for the minority class (death), which is unacceptable.

| Metric | Imbalanced Dataset (Initial) | Balanced Dataset (Optimized) |
| :--- | :---: | :---: |
| **Accuracy** | ~92% (Misleading) | ~79% |
| **Recall (Minority)** | Low | **High** |
| **Precision** | Moderate | High |
| **Generalization** | Prone to Overfitting | Robust |

> **Conclusion**: Balancing the dataset and applying L2 Regularization significantly improved the model's ability to identify high-risk patients, increasing the F1-Score for the minority class.
