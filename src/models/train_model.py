import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns 

# Plot settings
plt.style.use("fivethirtyeight")
plt. rcParams ["figure.figsize"]
plt. rcParams ["figure.dpi"] = 100
plt. rcParams ["lines.linewidth"] = 2
(20, 5)

df = pd. read_pickle("/Users/bhaveshmankar/data-science-template/data/interim/03_data_features.pkl")

column_list = list(df.columns)
print(column_list)

df_train = df.drop( ["participant", "category", "set"], axis=1)
X = df_train.drop("label", axis=1)
y = df_train ["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
fig, ax = plt.subplots(figsize=(10, 5))
df_train ["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
y_test. value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test" )
plt.legend()
plt.show()


basic_features = ["acc_x",
"acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features =["ace_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_ 3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_features = ["cluster"]
print("Basic features:", len(basic_features))
print("Square featuses:", len(square_features))
print("PCA features:",len (pca_features) )
print ("Time features:", len(time_features) )
print ("Frequency features:", len(freq_features))
print("Cluster features:", len(cluster_features))



clf = DecisionTreeClassifier(
    criterion="entropy",    # or "entropy"
    max_depth=None,      # you can set e.g. 5 or None for no limit
    min_samples_leaf=6,  # prevents overfitting
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 3) Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_,
    cmap="Blues",
    ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 4) Visualize the tree (first 3 levels)
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
