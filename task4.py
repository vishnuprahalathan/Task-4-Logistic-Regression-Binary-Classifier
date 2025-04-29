
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


csv_path = r"C:\Users\Vishnu Prahalathan\Desktop\data.csv"
df = pd.read_csv(csv_path)


df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'unnamed' in col.lower()], inplace=True)

if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


df.dropna(inplace=True)


X = df.drop(columns='diagnosis')
y = df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 Confusion Matrix:\n", conf_matrix)
print("\n📋 Classification Report:\n", report)
print("🎯 ROC-AUC Score: {:.4f}".format(roc_auc))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 200)
sigmoid_vals = sigmoid(z)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(z, sigmoid_vals, color='green')
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()
