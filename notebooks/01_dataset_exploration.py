import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining dataset columns:")
print(train_df.columns)

print("\nFirst five records:")
print(train_df.head())
print("\nNumber of unique categories:")
print(train_df["category"].nunique())

print("\nCategory distribution:")
print(train_df["category"].value_counts())

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nDuplicate records:")
print(train_df.duplicated().sum())
import matplotlib.pyplot as plt

category_counts = train_df["category"].value_counts()

plt.figure(figsize=(16, 6))
#category_counts.plot(kind="bar")

plt.title("Distribution of Banking Intent Categories")
plt.xlabel("Categories")
plt.ylabel("Number of Queries")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
print("\nSample customer queries:")

for text in train_df["text"].sample(10, random_state=42):
    print("-", text)
print("Empty queries:", train_df["text"].str.strip().eq("").sum())

print("Duplicate queries:", train_df["text"].duplicated().sum())

print("Missing values:", train_df.isnull().sum())
X = train_df["text"]
y = train_df["category"]

print("Input shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X)

print("TF-IDF shape:", X_train_tfidf.shape)
print("Total vocabulary:", len(vectorizer.vocabulary_))

print("\nFirst 30 words:")
print(list(vectorizer.vocabulary_.keys())[:30])
sample_text = [X.iloc[0]]

sample_vector = vectorizer.transform(sample_text)

print("Original query:")
print(sample_text[0])

print("\nNumerical representation:")
print(sample_vector.toarray())
print("Non-zero values:")
print(sample_vector.nonzero())

print("\nNumber of non-zero values:")
print(sample_vector.nnz)
feature_names = vectorizer.get_feature_names_out()
values = sample_vector.toarray()[0]

for index in sample_vector.nonzero()[1]:
    print(feature_names[index], ":", values[index])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(C=5.0, max_iter=1000))
])
model.fit(train_df["text"], train_df["category"])
print("Model training completed!")
from sklearn.metrics import accuracy_score, classification_report

# Predict categories for unseen queries
y_pred = model.predict(test_df["text"])

# Compare predictions with actual categories
accuracy = accuracy_score(test_df["category"], y_pred)

print("Model Accuracy:", accuracy)
# print("\nClassification Report:")
# print(classification_report(test_df["category"], y_pred))
# results = test_df.copy()

# results["predicted_category"] = y_pred

# incorrect = results[
#     results["category"] != results["predicted_category"]
# ]

# print("Total incorrect predictions:", len(incorrect))

# print("\nFirst 15 incorrect predictions:")

# print(
#     incorrect[
#         ["text", "category", "predicted_category"]
#     ].head(15).to_string(index=False)
# )\
# Training accuracy
train_predictions = model.predict(train_df["text"])
train_accuracy = accuracy_score(
    train_df["category"],
    train_predictions
)

# Testing accuracy
test_predictions = model.predict(test_df["text"])
test_accuracy = accuracy_score(
    test_df["category"],
    test_predictions
)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
import pandas as pd

results = test_df.copy()

results["predicted_category"] = y_pred

incorrect = results[
    results["category"] != results["predicted_category"]
]

confusion_pairs = (
    incorrect.groupby(
        ["category", "predicted_category"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(confusion_pairs.head(15).to_string(index=False))
import joblib

joblib.dump(model, "models/bankassist_model.pkl")

print("BankAssist model saved successfully!")
