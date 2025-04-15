import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Load data
df = pd.read_pickle("/Users/bhaveshmankar/data-science-template/data/interim/03_data_features.pkl")

# Prepare data
df_train = df.drop(["participant", "category", "set"], axis=1)
X = df_train.drop("label", axis=1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ========== Decision Tree Classifier ==========
print("\n================ Decision Tree ================\n")
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=6,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree - Test Accuracy: {acc_dt:.3f}\n")
print("Decision Tree - Classification Report:")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Decision Tree - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    max_depth=3,
    feature_names=X_train.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax,
)
plt.title("Decision Tree (first 3 levels)")
plt.show()


print("\n================ Random Forest ================\n")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=6,
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest - Test Accuracy: {acc_rf:.3f}\n")
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_clf.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=rf_clf.classes_,
    yticklabels=rf_clf.classes_,
    cmap="Greens",
    ax=ax,
)
ax.set_title("Random Forest - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.show()


