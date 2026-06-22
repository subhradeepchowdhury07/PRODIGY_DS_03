# ============================================
# PRODIGY TASK-03
# Decision Tree Classifier - Bank Marketing
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("bank-additional.csv", sep=";")

print("=" * 50)
print("BANK MARKETING DATASET")
print("=" * 50)

print("\nFirst 5 Rows")
print(df.head())

print("\nShape of Dataset")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())

# ============================================
# Encode Categorical Columns
# ============================================

label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = label_encoder.fit_transform(df[column])

# ============================================
# Features and Target
# ============================================

X = df.drop("y", axis=1)
y = df["y"]

# ============================================
# Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================
# Train Decision Tree
# ============================================

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ============================================
# Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Evaluation
# ============================================

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ============================================
# Plot Decision Tree
# ============================================

plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Bank Marketing")
plt.savefig("decision_tree.png")
plt.show()

# ============================================
# Plot Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No", "Yes"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nDecision Tree Image saved as decision_tree.png")
print("Confusion Matrix saved as confusion_matrix.png")
print("\nTask Completed Successfully!")
