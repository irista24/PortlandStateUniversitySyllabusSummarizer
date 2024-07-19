from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import io
import re
import pdfplumber
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn

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

# Initialize DistilBERT model
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

# Extract text from PDF file
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove newlines within sentences
    text = re.sub(r'\n+', ' ', text)
    # Fix punctuation spacing
    text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
    return text

def is_valid_sentence(sentence):
    # Example of a simple validation, can be expanded as needed
    return len(sentence.strip()) > 0  # Ensure sentence is not empty after stripping

# Break text into sentences
def break_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# Format and clean sentences
def format_sentences(sentences):
    formatted_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces
        sentence = sentence.strip()
        if sentence:
            formatted_sentences.append(sentence)
    return formatted_sentences

# Match sentences with keywords
def match_sentences_with_keywords(sentences, keywords):
    matched_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                matched_sentences.append((keyword, sentence))
                break
    return matched_sentences

# Store matched sentences in database
def store_matched_sentences_in_db(syllabus_id, matched_sentences):
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    for keyword, sentence in matched_sentences:
        c.execute("INSERT INTO syllabi (syllabus_id, keyword, sentence) VALUES (?, ?, ?)", (syllabus_id, keyword, sentence))
    conn.commit()
    conn.close()

# Upload PDF endpoint
@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        text = extract_text_from_pdf(file)
        cleaned_text = clean_text(text)
        sentences = break_text_into_sentences(cleaned_text)
        formatted_sentences = format_sentences(sentences)
        matched_sentences = match_sentences_with_keywords(formatted_sentences, keywords)

        conn = sqlite3.connect('syllabi.db')
        c = conn.cursor()
        c.execute("SELECT MAX(syllabus_id) FROM syllabi")
        result = c.fetchone()
        new_syllabus_id = result[0] + 1 if result[0] is not None else 1
        conn.close()

        store_matched_sentences_in_db(new_syllabus_id, matched_sentences)
        return jsonify({"message": "File uploaded and processed", "syllabus_id": new_syllabus_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Summarization endpoint
@app.route('/summary', methods=['POST'])
def generate_summary():
    data = request.json
    syllabus_id = data.get('syllabus_id')

    if not syllabus_id:
        return jsonify({"error": "No syllabus_id provided"}), 400

    try:
        conn = sqlite3.connect('syllabi.db')
        df = pd.read_sql_query(f"SELECT sentence FROM syllabi WHERE syllabus_id = {syllabus_id}", conn)
        conn.close()

        summary = "\n".join(df['sentence'].tolist())
        return jsonify({"summary": summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Answering questions endpoint
@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        conn = sqlite3.connect('syllabi.db')
        c = conn.cursor()
        c.execute("SELECT MAX(syllabus_id) FROM syllabi")
        result = c.fetchone()
        recent_syllabus_id = result[0]

        if recent_syllabus_id is None:
            return jsonify({"answer": "No syllabus data found."}), 200

        df = pd.read_sql_query(f"SELECT * FROM syllabi WHERE syllabus_id = {recent_syllabus_id}", conn)
        conn.close()

        answer_parts = []

        for index, row in df.iterrows():
            keyword = row['keyword']
            sentence = row['sentence']
            
            # Clean and validate the sentence
            cleaned_sentence = clean_text(sentence)
            if is_valid_sentence(cleaned_sentence) and keyword.lower() in question.lower():
                answer_parts.append(cleaned_sentence)

        if answer_parts:
            # Join the sentences with proper spacing
            formatted_answer = " ".join(answer_parts)
            return jsonify({"answer": formatted_answer}), 200
        else:
            return jsonify({"answer": "No relevant information found."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to search for keyword instances
@app.route('/search_keyword', methods=['POST'])
def search_keyword():
    data = request.json
    keyword = data.get('keyword')

    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    try:
        # Load the CSV file containing the trained data
        df = pd.read_csv('/u/irist_guest/Desktop/pdfs/aw.csv')
        keyword_lower = keyword.lower()

        matched_sentences = []

        for index, row in df.iterrows():
            text = row['Text']

            if keyword_lower in text.lower():
                matched_sentences.append(text)

        return jsonify({"instances": matched_sentences}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def welcome():
    return "Welcome to the Syllabus Q&A API!"

if __name__ == '__main__':
    init_db()
    init_model()
    app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3
# import io
# import re
# import pdfplumber
# import pandas as pd
# from transformers import DistilBertTokenizer, DistilBertModel
# import torch
# import torch.nn as nn

# app = Flask(__name__)
# CORS(app)

# # Initialize tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = None

# # Define keywords
# keywords = ['Instructor', 'Email', 'Office', 'Late Work', 'Mentor', 'Course Description', 'Objective', 'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 'Attendance', 'Academic Integrity', 'Peer Mentor', 'Technology']

# # Create SQLite database
# def init_db():
#     conn = sqlite3.connect('syllabi.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS syllabi
#                  (id INTEGER PRIMARY KEY, syllabus_id INTEGER, keyword TEXT, sentence TEXT)''')
#     conn.commit()
#     conn.close()

# # Initialize DistilBERT model
# class LabelModel(nn.Module):
#     def __init__(self, num_labels):
#         super(LabelModel, self).__init__()
#         self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
#         self.dropout = nn.Dropout(0.3)
#         self.output = nn.Linear(768, num_labels)
    
#     def forward(self, ids, mask):
#         output = self.distilbert(ids, attention_mask=mask)
#         output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
#         output = self.output(output)
#         return output

# def init_model():
#     global model
#     # Load state dict to get the number of labels
#     state_dict = torch.load('/u/irist_guest/Desktop/pdfs/model.pth', map_location=torch.device('cpu'))
#     num_labels = state_dict['output.weight'].size(0)
#     model = LabelModel(num_labels)
#     model.load_state_dict(state_dict)
#     model.eval()

# # Extract text from PDF file
# def extract_text_from_pdf(file):
#     text = ""
#     try:
#         with pdfplumber.open(io.BytesIO(file.read())) as pdf:
#             for page in pdf.pages:
#                 text += page.extract_text()
#     except Exception as e:
#         print(f"Error extracting text from PDF: {e}")
#     return text


# def clean_text(text):
#     # Remove excessive whitespace
#     text = re.sub(r'\s+', ' ', text)
#     # Remove leading and trailing whitespace
#     text = text.strip()
#     # Remove newlines within sentences
#     text = re.sub(r'\n+', ' ', text)
#     # Fix punctuation spacing
#     text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
#     return text

# def is_valid_sentence(sentence):
#     # Example of a simple validation, can be expanded as needed
#     return len(sentence.strip()) > 0  # Ensure sentence is not empty after stripping
# # Break text into sentences
# def break_text_into_sentences(text):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     return sentences

# # Format and clean sentences
# def format_sentences(sentences):
#     formatted_sentences = []
#     for sentence in sentences:
#         sentence = re.sub(r'\s+', ' ', sentence)  # Remove multiple spaces
#         sentence = sentence.strip()
#         if sentence:
#             formatted_sentences.append(sentence)
#     return formatted_sentences

# # Match sentences with keywords
# def match_sentences_with_keywords(sentences, keywords):
#     matched_sentences = []
#     for sentence in sentences:
#         for keyword in keywords:
#             if keyword.lower() in sentence.lower():
#                 matched_sentences.append((keyword, sentence))
#                 break
#     return matched_sentences

# # Store matched sentences in database
# def store_matched_sentences_in_db(syllabus_id, matched_sentences):
#     conn = sqlite3.connect('syllabi.db')
#     c = conn.cursor()
#     for keyword, sentence in matched_sentences:
#         c.execute("INSERT INTO syllabi (syllabus_id, keyword, sentence) VALUES (?, ?, ?)", (syllabus_id, keyword, sentence))
#     conn.commit()
#     conn.close()

# # Upload PDF endpoint
# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     try:
#         text = extract_text_from_pdf(file)
#         cleaned_text = clean_text(text)
#         sentences = break_text_into_sentences(cleaned_text)
#         formatted_sentences = format_sentences(sentences)
#         matched_sentences = match_sentences_with_keywords(formatted_sentences, keywords)

#         conn = sqlite3.connect('syllabi.db')
#         c = conn.cursor()
#         c.execute("SELECT MAX(syllabus_id) FROM syllabi")
#         result = c.fetchone()
#         new_syllabus_id = result[0] + 1 if result[0] is not None else 1
#         conn.close()

#         store_matched_sentences_in_db(new_syllabus_id, matched_sentences)
#         return jsonify({"message": "File uploaded and processed", "syllabus_id": new_syllabus_id}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Summarization endpoint
# @app.route('/summary', methods=['POST'])
# def generate_summary():
#     data = request.json
#     syllabus_id = data.get('syllabus_id')

#     if not syllabus_id:
#         return jsonify({"error": "No syllabus_id provided"}), 400

#     try:
#         conn = sqlite3.connect('syllabi.db')
#         df = pd.read_sql_query(f"SELECT sentence FROM syllabi WHERE syllabus_id = {syllabus_id}", conn)
#         conn.close()

#         summary = "\n".join(df['sentence'].tolist())
#         return jsonify({"summary": summary}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Answering questions endpoint
# @app.route('/answer', methods=['POST'])
# def answer_question():
#     data = request.json
#     question = data.get('question')

#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         conn = sqlite3.connect('syllabi.db')
#         c = conn.cursor()
#         c.execute("SELECT MAX(syllabus_id) FROM syllabi")
#         result = c.fetchone()
#         recent_syllabus_id = result[0]

#         if recent_syllabus_id is None:
#             return jsonify({"answer": "No syllabus data found."}), 200

#         df = pd.read_sql_query(f"SELECT * FROM syllabi WHERE syllabus_id = {recent_syllabus_id}", conn)
#         conn.close()

#         answer_parts = []

#         for index, row in df.iterrows():
#             keyword = row['keyword']
#             sentence = row['sentence']
            
#             # Clean and validate the sentence
#             cleaned_sentence = clean_text(sentence)
#             if is_valid_sentence(cleaned_sentence) and keyword.lower() in question.lower():
#                 answer_parts.append(cleaned_sentence)

#         if answer_parts:
#             # Join the sentences with proper spacing
#             formatted_answer = " ".join(answer_parts)
#             return jsonify({"answer": formatted_answer}), 200
#         else:
#             return jsonify({"answer": "No relevant information found."}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

    
# @app.route('/')
# def welcome():
#     return "Welcome to the Syllabus Q&A API!"

# if __name__ == '__main__':
#     init_db()
#     init_model()
#     app.run(debug=True)