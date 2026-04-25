# Fake News Detection Project
# Author: Mehek Zahra Ajmal

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Select features and target
X = data["text"]
y = data["label"]

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy, 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to test custom news
def predict_news(news_text):
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)

    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"

# Example test
sample_news = "Scientists discover new method to treat cancer"
print("\nSample Prediction:", predict_news(sample_news))