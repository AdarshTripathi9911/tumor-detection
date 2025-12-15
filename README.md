# tumor-detection
Tumor Detection Project  This project uses machine learning to classify breast tumors as malignant or benign using the sklearn breast cancer dataset. A Random Forest Classifier was trained and achieved high accuracy (~97–98%). Feature importance analysis highlighted key tumor characteristics, demonstrating that ML can support early tumor detection.




# Tumor Detection Project - Using sklearn Breast Cancer Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='diagnosis')  # 0 = malignant, 1 = benign

# Combine X and y for EDA
df = pd.concat([X, y], axis=1)

# -----------------------------
# 2. Exploratory Data Analysis (EDA)
# -----------------------------
# Diagnosis distribution
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df)
plt.title("Diagnosis Distribution (0=Malignant, 1=Benign)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions & Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 7. Feature Importance
# -----------------------------
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features for Tumor Prediction")
plt.show()
