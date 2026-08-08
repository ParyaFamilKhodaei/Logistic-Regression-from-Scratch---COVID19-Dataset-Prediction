# Logistic_Regression_COVID19_Dataset_Prediction_from_scratch
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
- **From-Scratch Implementation:** Logistic Regression, Sigmoid activation, Binary Cross-Entropy loss, Gradient computation, and Gradient Descent are implemented without using Scikit-Learn's `LogisticRegression`.
- **Mathematical Foundation:** Complete manual implementation of the learning algorithm.
- **Imbalance Analysis:** Comparison of model performance on both the original imbalanced dataset and a balanced dataset.
- **Overfitting Evaluation:** Training and testing costs are compared to determine whether the model suffers from overfitting and whether regularization would be beneficial.

---

## 🛠 Technical Implementation

### 1. Data Preprocessing Pipeline

The raw clinical data required preprocessing before training:

- **Missing Value Handling:** Removal of null values and placeholder characters (`#`).
- **Feature Engineering:** Conversion of the `date_died` column into a binary target variable (`1 = deceased`, `0 = survived`).
- **Categorical Encoding:** Encoding categorical features using `LabelEncoder`.
- **Feature Scaling:** Manual Min-Max normalization using statistics computed from the training set only.

---

### 2. Mathematics of the Model

The model minimizes the Binary Cross-Entropy loss function:

$$
J(w,b)=
-\frac1m
\sum_{i=1}^{m}
\left[
y^{(i)}\log(f_{w,b}(x^{(i)}))
+
(1-y^{(i)})\log(1-f_{w,b}(x^{(i)}))
\right]
$$

where the hypothesis function is

$$
f_{w,b}(x)=\frac1{1+e^{-(wx+b)}}
$$

Gradient Descent is then used to iteratively optimize the model parameters.

---

### 3. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

These metrics are computed using **Scikit-Learn's evaluation functions**, while the Logistic Regression algorithm itself is implemented entirely from scratch.

Training and testing costs are also compared to evaluate the model's ability to generalize and to determine whether regularization may be required.

---

## 📊 Comparative Analysis: Imbalanced vs. Balanced Data

The project investigates the **Accuracy Paradox** commonly encountered in medical datasets. A classifier may achieve high overall accuracy while performing poorly on the minority class (deceased patients).

The model is therefore trained and evaluated on:

- **Original (Imbalanced) Dataset**
- **Balanced Dataset**

Performance is compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Training Cost
- Testing Cost

This comparison demonstrates how balancing the dataset influences the model's ability to identify minority-class samples and improves overall generalization.

---

## 📈 Visualizations

The project includes plots of the training cost versus iterations for both datasets, allowing visualization of Gradient Descent convergence and helping assess the optimization process.
