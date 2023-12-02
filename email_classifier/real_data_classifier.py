import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
dataset_path = './my_email/Emails.csv'  # Replace with the actual path
loaded_data = pd.read_csv(dataset_path)

# Split data into testing and training sets
train_data, test_data = train_test_split(loaded_data, test_size=0.2, random_state=42)

# Train the model
vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(train_data['text'].values)
train_targets = train_data['spam'].values
classifier = MultinomialNB()
classifier.fit(train_counts, train_targets)

# Test the model
test_counts = vectorizer.transform(test_data['text'].values)
test_targets = test_data['spam'].values
predictions = classifier.predict(test_counts)

# Evaluate accuracy
accuracy = accuracy_score(test_targets, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(test_targets, predictions))
