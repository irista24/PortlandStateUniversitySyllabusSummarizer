from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import io
from PyPDF2 import PdfReader
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import re

app = Flask(__name__)
CORS(app)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = None

# Define keywords
keywords = ['Instructor', 'Email', 'Office', 'Late Work', 'Mentor', 'Course Description', 'Objective', 'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 'Attendance', 'Academic Integrity', 'Peer Mentor', 'Technology']

# Create SQLite database
def init_db():
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS syllabi
                 (id INTEGER PRIMARY KEY, syllabus_id INTEGER, keyword TEXT, sentence TEXT)''')
    conn.commit()
    conn.close()


class LabelModel(nn.Module):
    def __init__(self, num_labels):
        super(LabelModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(768, num_labels)
    
    def forward(self, ids, mask):
        output = self.distilbert(ids, attention_mask=mask)
        output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
        output = self.output(output)
        return output

def init_model():
    global model
    # Load state dict to get the number of labels
    state_dict = torch.load('/u/irist_guest/Desktop/pdfs/model.pth', map_location=torch.device('cpu'))
    num_labels = state_dict['output.weight'].size(0)
    model = LabelModel(num_labels)
    model.load_state_dict(state_dict)
    model.eval()

from PyPDF2 import PdfReader
import io

def extract_text_from_pdf(file):
    pdf = PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Fix common formatting issues
    text = re.sub(r'\n([a-z])', r' \1', text)  # Fix newlines within sentences
    text = re.sub(r'([.!?])\s*\n', r'\1 ', text)  # Fix newlines after punctuation

    # Additional custom cleaning rules can be added here

    return text


# Break text into sentences
def break_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# Match sentences with keywords
def match_sentences_with_keywords(sentences, keywords):
    matched_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                matched_sentences.append((keyword, sentence))
                break
    return matched_sentences

def store_matched_sentences_in_db(syllabus_id, matched_sentences):
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    for keyword, sentence in matched_sentences:
        c.execute("INSERT INTO syllabi (syllabus_id, keyword, sentence) VALUES (?, ?, ?)", (syllabus_id, keyword, sentence))
    conn.commit()
    conn.close()

@app.route('/upload', methods=['POST'])
def upload_pdf():
    print("Upload endpoint hit")
    if 'file' not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    text = extract_text_from_pdf(file)
    text = clean_text(text)  # Clean the extracted text
    print("Extracted text:", text[:100])  # Print the first 100 characters of the extracted text
    sentences = break_text_into_sentences(text)
    print("First few sentences:", sentences[:3])  # Print the first few sentences
    matched_sentences = match_sentences_with_keywords(sentences, keywords)
    print("Matched sentences:", matched_sentences[:3])  # Print the first few matched sentences

    # Generate a new syllabus_id
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute("SELECT MAX(syllabus_id) FROM syllabi")
    result = c.fetchone()
    if result[0] is None:
        new_syllabus_id = 1
    else:
        new_syllabus_id = result[0] + 1
    conn.close()

    store_matched_sentences_in_db(new_syllabus_id, matched_sentences)
    return jsonify({"message": "File uploaded and processed", "syllabus_id": new_syllabus_id}), 200


# Route to handle question answering
@app.route('/answer', methods=['POST'])
def answer_question():
    print("Answer endpoint hit")
    data = request.json
    print("Received question:", data['question'])
    question = data['question']

    # Find the most recent syllabus_id
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute("SELECT MAX(syllabus_id) FROM syllabi")
    result = c.fetchone()
    if result[0] is None:
        return jsonify({"answer": "No syllabi found."}), 200

    recent_syllabus_id = result[0]

    # Fetch sentences from the most recent syllabus
    df = pd.read_sql_query(f"SELECT * FROM syllabi WHERE syllabus_id = {recent_syllabus_id}", conn)
    conn.close()
    print("Data from DB:", df.head())

    # Find the most relevant sentence
    best_match = None
    highest_similarity = 0

    for index, row in df.iterrows():
        keyword = row['keyword']
        sentence = row['sentence']
        if keyword.lower() in question.lower():
            best_match = sentence
            break

    if best_match:
        print("Best match found:", best_match)
        return jsonify({"answer": best_match}), 200
    else:
        print("No relevant information found")
        return jsonify({"answer": "No relevant information found."}), 200

@app.route('/')
def index():
    return "Welcome to the Syllabus QA API"


if __name__ == '__main__':
    init_db()
    init_model()
    app.run(debug=True)