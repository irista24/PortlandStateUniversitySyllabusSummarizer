from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from PyPDF2 import PdfFileReader
import io
import pandas as pd
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS syllabi
                 (id INTEGER PRIMARY KEY, label TEXT, text TEXT)''')
    conn.commit()
    conn.close()

# Load tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define the Dataset class
class SyllabiDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row["Text"]
        label = row["Label"]
        
        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.max_len, padding="max_length", return_token_type_ids=True, truncation=True)
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long), 
            "targets": torch.tensor(label, dtype=torch.long)
        }

# Define the Model class
class LabelModel(nn.Module):
    def __init__(self, num_labels):
        super(LabelModel, self).__init__()
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, num_labels)

    def forward(self, ids, mask):
        output = self.distilbert(ids, attention_mask=mask)
        output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
        output = self.output(output)
        return output

# Initialize the model
def init_model():
    global model
    conn = sqlite3.connect('syllabi.db')
    df = pd.read_csv('/u/irist_guest/Desktop/pdfs/aw.csv')
    num_labels = len(df["Label"].unique())
    model = LabelModel(num_labels)
    model.load_state_dict(torch.load('/u/irist_guest/Desktop/pdfs/model.pth', map_location=torch.device('cpu')))
    model.eval()

from PyPDF2 import PdfReader  # Update the import

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf = PdfReader(io.BytesIO(pdf_file.read()))  # Use PdfReader instead of PdfFileReader
    text = ""
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
    return text
# Root route to handle the root URL
@app.route('/')
def index():
    return "Welcome to the Syllabus Q&A and Summarization API!"

# Route to handle PDF uploads
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    text = extract_text_from_pdf(file)
    label = "default"  # Placeholder for the label, adjust as needed
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute("INSERT INTO syllabi (label, text) VALUES (?, ?)", (label, text))
    conn.commit()
    conn.close()
    return jsonify({"message": "File uploaded and processed"}), 200

# Route to handle question answering
@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.json
    question = data['question']
    conn = sqlite3.connect('syllabi.db')
    df = pd.read_sql_query("SELECT * FROM syllabi", conn)
    conn.close()

    texts = df["text"].tolist()
    question_embedding = get_embeddings([question], tokenizer, model)
    text_embeddings = get_embeddings(texts, tokenizer, model)

    similarity_scores = cosine_similarity(question_embedding, text_embeddings)
    most_similar_index = similarity_scores.argmax()
    most_similar_text = texts[most_similar_index]

    return jsonify({"answer": most_similar_text}), 200


# Route to handle syllabus summary
@app.route('/summary', methods=['POST'])
def summarize_syllabus():
    conn = sqlite3.connect('syllabi.db')
    df = pd.read_sql_query("SELECT * FROM syllabi", conn)
    conn.close()
    
    # Simple summary logic, can be improved with more sophisticated techniques
    summary = " ".join(df["text"].tolist()[:5])  # Concatenate first 5 texts as a simple summary

    return jsonify({"summary": summary}), 200

if __name__ == '__main__':
    init_db()
    init_model()
    app.run(debug=True)
