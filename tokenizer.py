import nltk 
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt')
def tokenize_file(file_path): 
    with open(file_path, 'r', encoding = "utf-8") as file: 
        text1 = file.read()
        tokens = word_tokenize(text1)
    return tokens
def tokenize_files_in_directory(pdfs):
    tokens_dict = {}
    for filename in os.listdir(pdfs):
        if filename.endswith('.txt'):
            file_path = os.path.join(pdfs, filename)
            tokens = tokenize_file(file_path)
            tokens_dict[filename] = tokens
    return tokens_dict
directory_path = '/u/irist_guest/Desktop/pdfs/'
tokenized_files = tokenize_files_in_directory(directory_path)
