# Email Classifier

This is a Python program that uses the Naive Bayes algorithm to classify emails as 'spam' or 'ham'. The program reads emails from files, preprocesses the emails, trains a Multinomial Naive Bayes classifier on the data, makes predictions on a sample of emails, and evaluates the accuracy of the model on a test set.

## Getting Started

### Prerequisites

You need to have Python 3.6 or higher installed on your machine. You also need to install the following Python libraries:

- pandas
- numpy
- sklearn
- <img width="370" alt="Screenshot 2023-12-01 at 11 58 32 PM" src="https://github.com/sumtzehern/spamHam/assets/77678835/ac66199a-e0d6-4fc8-8618-eecfb09d3d2d">


## Data Preprocessing
The program reads emails from files in a given directory. It assumes that the emails are separated into 'spam' and 'ham' directories. The program reads each file line by line and collects the lines that are part of the email body.

The program then creates a pandas DataFrame where each row represents an email and has two columns: 'message' (the email content) and 'class' (the classification of the email).

### Model Training
The program uses the CountVectorizer class from sklearn to convert the email texts into a matrix of token counts. It then trains a Multinomial Naive Bayes classifier on the prepared data.

### Making Predictions
The program tests the classifier on a sample of emails. It first transforms the sample into a matrix of token counts using the same vectorizer that was used to prepare the training data. It then uses the classifier to predict the class of each email and to calculate the probabilities of each class.

### Evaluating the Model
The program splits the data into a training set and a test set. It trains the model on the training set and tests it on the test set. It then calculates the accuracy of the model on the test set.

### Things to be Aware Of
The program assumes that the emails are separated into 'spam' and 'ham' directories. If your data is organized differently, you will need to modify the program accordingly.
The program uses the CountVectorizer class from sklearn to convert the email texts into a matrix of token counts. This class has several parameters that control how the text is preprocessed. You might need to adjust these parameters depending on your data.
The program uses the Multinomial Naive Bayes classifier from sklearn for classification. This class has several hyperparameters that you might want to tune to improve the performance of the model.
The program calculates the accuracy of the model on the test set. While accuracy is a good metric, it might not be sufficient for all problems. Depending on the distribution of your data, you might want to calculate other metrics like precision, recall, F1 score, or AUC-ROC.

## Results
The program was tested on a dataset of emails for spam detection from Kaggle. The dataset was split into a training set and a test set, and the Multinomial Naive Bayes classifier was trained on the training set. The accuracy of the model on the test set was 98.78%. The classification report for the model is as follows:
<img width="477" alt="Screenshot 2023-12-01 at 11 58 21 PM" src="https://github.com/sumtzehern/spamHam/assets/77678835/93603e94-8de8-4e5b-8bb7-9cd359b54305">
<img width="550" alt="Screenshot 2023-12-01 at 11 58 00 PM" src="https://github.com/sumtzehern/spamHam/assets/77678835/a0ea8a3d-bca2-4283-8105-804073d5bf39">
