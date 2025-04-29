# Task-4-Logistic-Regression-Binary-Classifier


This project is part of the AI & ML Internship and demonstrates a complete workflow for solving a **binary classification problem using Logistic Regression on the Breast Cancer Wisconsin dataset.

---

Objective

Build and evaluate a binary classification model that predicts whether a tumor is **malignant (M)** or benign (B) based on diagnostic features.

---

Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

Dataset

- File: `data.csv`
- Source: Breast Cancer Wisconsin Dataset
- Target Column: `diagnosis`  
  - M → 1 (Malignant)  
  - B → 0 (Benign)

---

Workflow

1. Load & Clean Data
   - Drop unnecessary columns like `id` and unnamed ones
   - Convert `diagnosis` from categorical to numerical
   - Drop rows with missing values

2. Preprocessing
   - Train-test split (80/20)
   - Standardize features using `StandardScaler`

3. Model Training
   - Logistic Regression (Scikit-learn)

4. Evaluation
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC Score
   - ROC Curve

5. Visualization
   - Sigmoid function
   - ROC curve

---

Results

- Model Accuracy:~97%
- ROC-AUC Score: ~0.99
- Balanced Precision and Recall

---

