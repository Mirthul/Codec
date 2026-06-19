import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

import shap

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("loan_data.csv", encoding="latin1")

print("\nDataset Preview")
print(df.head())

print("\nColumns")
print(df.columns.tolist())

# ==================================================
# HANDLE MISSING VALUES
# ==================================================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ==================================================
# TARGET COLUMN
# ==================================================

target = "Status of existing checking account"

# ==================================================
# ENCODE CATEGORICAL COLUMNS
# ==================================================

label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df.drop(columns=[target])

y = df[target]

print("\nTarget Distribution Before SMOTE")
print(y.value_counts())

# ==================================================
# APPLY SMOTE
# ==================================================

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nTarget Distribution After SMOTE")
print(pd.Series(y_resampled).value_counts())

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# ==================================================
# LOGISTIC REGRESSION
# ==================================================

print("\n==========================")
print("LOGISTIC REGRESSION")
print("==========================")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, lr_pred)*100,2),
      "%")

# ==================================================
# DECISION TREE
# ==================================================

print("\n==========================")
print("DECISION TREE")
print("==========================")

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, dt_pred)*100,2),
      "%")

# ==================================================
# RANDOM FOREST
# ==================================================

print("\n==========================")
print("RANDOM FOREST")
print("==========================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Accuracy:",
      round(accuracy_score(y_test, rf_pred)*100,2),
      "%")

# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")

print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance.head(10)
)

plt.title("Top 10 Important Features")
plt.show()

# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

print("\nGenerating SHAP Analysis...")

explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test
)

# ==================================================
# SAMPLE PREDICTION
# ==================================================

sample = X_test.iloc[[0]]

prediction = rf.predict(sample)

print("\nSample Prediction Result")

if prediction[0] == 1:
    print("Credit Risk: LOW")
    print("Loan Likely Approved")
else:
    print("Credit Risk: HIGH")
    print("Loan Likely Rejected")

print("\nProject Completed Successfully")
