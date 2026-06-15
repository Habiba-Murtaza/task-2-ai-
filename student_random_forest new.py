
# Random Forest Classifier - Student Dataset


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("student_academic_data.csv")
print("Dataset Shape:", df.shape)
print(df.head())


# 2. PREPROCESSING


# Encode Gender (Male=1, Female=0)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# Encode Status (Pass=1, Fail=0) → Target variable
df["Status"] = df["Status"].map({"Pass": 1, "Fail": 0})

# Drop non-useful columns
df.drop(columns=["StudentID", "Name"], inplace=True)

print("\nProcessed Data:")
print(df.head())


# 3. DEFINE FEATURES & TARGET

X = df.drop(columns=["Status"])   # Features
y = df["Status"]                   # Target

print("\nFeatures:", list(X.columns))
print("Target Distribution:\n", y.value_counts())


# 4. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")


# 5. TRAIN RANDOM FOREST MODEL

model = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=5,        # Max depth of each tree
    random_state=42
)
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")


# 6. PREDICTIONS & EVALUATION

y_pred = model.predict(X_test)

print("\n─── Model Evaluation ───")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))


# 7. CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# 8. FEATURE IMPORTANCE PLOT

importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature Importances:")
print(feat_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Feature importance plot saved as feature_importance.png")


# 9. PREDICT ON NEW STUDENT (Example)

new_student = pd.DataFrame([{
    "Age": 16,
    "Gender": 1,       # Male=1
    "Grade": 10,
    "Math": 72,
    "Science": 68,
    "English": 75,
    "History": 70,
    "Computer Science": 80,
    "Attendance (%)": 88,
    "GPA": 3.0
}])

prediction = model.predict(new_student)
result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"
print(f"\nPrediction for new student: {result}")
