import numpy as np
import pandas as pd
import re
import random
import torch
import torch.nn as nn 
import transformers 
import nltk 
from nltk.corpus import stopwords 
nltk.download("stopwords")
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_trivia
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size = 0.2, random_state=42)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

iris = load_trivia()
X = trivia.data
y = trivia.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
device = torch.device("cuda")

file = "/u/irist_guest/Desktop/pdfs/wy.csv"

df = pd.read_csv(file)
df.head()
df['Instructor'].value_counts()
le = LabelEncoder()
df["Instructor"] = le.fit_transform(df["Instructor"])
df["Instructor"].value_counts(normalize=True)


# Load the pre-trained BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained("distibert-baseuncased")

# Read the CSV file
csv_file_path = '/u/irist_guest/Desktop/pdfs/wy.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Specify the name of the text column
text_column = 'Instructor'  # Replace with the name of your text column

# Extract the text data
texts = df[text_column].tolist()

# Ensure texts are in the correct format (list of strings)
texts = [str(text) for text in texts]

# Tokenize the texts
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Print the tokenized output
print(encoded_inputs)

seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins=10)