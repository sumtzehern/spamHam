import os
import io
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Seperating the headers from body
def readFiles(path):
    """
    Reads emails from files in a given directory.
    
    Parameters:
    - path (str): Path to the directory containing email files.

    Yields:
    Tuple[str, str]: Tuple containing the file path and email body as a string.
    """
    for root, dirnames, filesnames in os.walk(path):
        for filename in filesnames:  #  iterate across files at 'path'
            path = os.path.join(root, filename)
            inBody = False
            lines = []  #  lines from email body will be saved.
            f = io.open(path, 'r', encoding='latin1')  # opening current file for reading. The 'r' param means read access.
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)  # goes through each string and combines into a big strink separated with spaces.
            yield path, message
            
def dataFrameFromDirectory(path, classification):
    """
    Creates a DataFrame from emails in files within a given directory.

    Parameters:
    - path (str): Path to the directory containing email files.
    - classification (str): Classification label for the emails (e.g., 'spam' or 'ham').

    Returns:
    pandas.DataFrame: DataFrame with 'message' and 'class' columns representing emails and their classifications.
    """
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({"message": message, 'class': classification})
        index.append(filename)
        
    return DataFrame(rows, index=index) # Takes two arrays 'rows'=emails, and 'index'=filenames

def save_to_folders(data, spam_folder_path, ham_folder_path):
    """
    Save messages to spam and ham folders based on their classifications.

    Parameters:
    - data (pandas.DataFrame): DataFrame with 'message' and 'class' columns representing emails and their classifications.
    - spam_folder_path (str): Path to the spam folder.
    - ham_folder_path (str): Path to the ham folder.
    """
    for index, row in data.iterrows():
        classification = row['class']
        message = row['message']
        folder_path = spam_folder_path if classification == 'spam' else ham_folder_path
        filename = os.path.join(folder_path, f"{index}.txt")

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(message)

# we are trying to have a column with the messages and a column that classifies the type of the message.
data = DataFrame({'message': [], 'class': []})

# Including the email details with the spam/ham classification in the dataframe
spam_folder_path = './full/spam'
ham_folder_path = './full/ham'
data = pd.concat([data, dataFrameFromDirectory(spam_folder_path, 'spam')])
data = pd.concat([data, dataFrameFromDirectory(ham_folder_path, 'ham')])

# Print the content of the data Frames, head and tail
print("DataFrame Head")
print(data.head())
print("\nDataFrame Tail")
print(data.tail())


#CountVectorizer is used to split up each message into its list of words
#Then we throw them to a MultinomialNB classifier function from scikit
#2 inputs required: actual data we are training on and the target data
vectorizer = CountVectorizer()

# vectorizer.fit_trsnform computes the word count in the emails and represents that as a frequency matrix (e.g., 'free' occured 1304 times.)
counts = vectorizer.fit_transform(data['message'].values)

#we will need to also have a list of ham/spam (corresponding to the emails from 'counts') that will allow Bayes Naive classifier compute the probabilities.
targets = data['class'].values

# This is from the sklearn package. MultinomialNB stands for Multinomial Naive Bayes classsifier
classifier = MultinomialNB()
# when we feed it the word frequencies plus the spam/ham mappings, the classifier will create a table of probabilities similar ot the one that you saw in the first assignment in this module.
classifier.fit(counts, targets)


'''MAKING PREDICTION'''
sample = ["Your order #SLNUSEN3753102 has been shipped",
          "SMC International Student Admission - Account Activation",
          "Update on your application",
          "[Action Required] Your Roblox Assessments Invitation"]

# 1. Transform the list into a table of word frequencies.
sample_counts = vectorizer.transform(sample)

# 2. ready to do the predictions.
predictions = classifier.predict(sample_counts)

print(sample,predictions)

proba = classifier.predict_proba(sample_counts)

for i, email in enumerate(sample):
    print(f"Email: {email}")
    print(f"Prediction: {predictions[i]}")
    print(f"Probability (ham, spam): {proba[i]}")
    print("\n")

'''ACCURACY TEST'''
# Split data to testing and training
# Use 80% of the data for training, and 20% for testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train the model on the training set
vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(train_data['message'].values)
train_targets = train_data['class'].values
classifier = MultinomialNB()
classifier.fit(train_counts, train_targets)

# Test the model on the testing set
test_counts = vectorizer.transform(test_data['message'].values)
test_targets = test_data['class'].values
predictions = classifier.predict(test_counts)

# Evaluate accuracy
accuracy = accuracy_score(test_targets, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Create text files to store spam and ham
spam_collection_file = open("spam_collection_email.txt", "w", encoding="utf-8")
ham_collection_file = open("ham_collection_email.txt", "w", encoding="utf-8")

# Iterate through the sample emails and write them to the respective text files
for i, email in enumerate(sample):
    if predictions[i] == 'spam':
        spam_collection_file.write(f"Email {i + 1}:\n{email}\n\n")
    else:
        ham_collection_file.write(f"Email {i + 1}:\n{email}\n\n")

# Close the text files
spam_collection_file.close()
ham_collection_file.close()

print("Sample emails have been classified and stored in 'spam_collection_email.txt' and 'ham_collection_email.txt'")

user_input = input("Do you want to categorize messages into different folders? (y/n): ").lower()

if user_input == 'y':
    # Define your spam and ham folders
    spam_folder_path = './full/spam'
    ham_folder_path = './full/ham'

    # Call the function to save messages to folders
    save_to_folders(data, spam_folder_path, ham_folder_path)
    print("Messages have been categorized into different folders.")
else:
    print("No categorization performed.")