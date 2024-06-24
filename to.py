import pandas as pd
from transformers import BertTokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
