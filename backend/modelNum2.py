from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import sqlite3
import io
import re
import pdfplumber
import concurrent.futures
import traceback
import pandas as pd
import transformers

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Load the model
# Load the tokenizer and model
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('/u/irist_guest/syllabus-app/x/cw.csv')
df = df[['Label', 'Text', 'Summary']]

class Label(torch.nn.Module):
    def __init__(self):
        super(Label, self).__init__()
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.output = torch.nn.Linear(768, len(df['Label'].unique()))

    def forward(self, ids, mask):
        output = self.distilbert(ids, attention_mask=mask)
        output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
        output = self.output(output)
        return output

# Initialize the model
model = Label()

# Load the saved model weights
model_path = '/u/irist_guest/syllabus-app/x/modelll.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.to(device)

# Load the label encoder
label_encoder = LabelEncoder()
# Example classes; you need to match this with your actual encoder classes
label_encoder.classes_ = np.array([ 'Instructor', 'Email', 'Office', 'Late Work', 'Course Description', 'Objective', 
    'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 
    'Attendance', 'Academic Integrity', 'Technology'])

# Load your dataframe

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Text"])

MAX_LEN = 256

# Initialize summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

# Define keywords
keywords = [
    'Instructor', 'Email', 'Office', 'Late Work', 'Course Description', 'Objective', 
    'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 
    'Attendance', 'Academic Integrity', 'Technology'
]

def init_db():
    conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
    c = conn.cursor()

    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS pdf_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS keyword_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS syllabi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            syllabus_id INTEGER,
            keyword TEXT,
            sentence TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE NOT NULL,
            summary TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-zA-Z]),([a-zA-Z])', r'\1, \2', text)
    text = re.sub(r'([a-zA-Z])\.([a-zA-Z])', r'\1. \2', text)
    text = text.replace('ﬁ', 'fi')
    return text

def break_text_into_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

def format_sentences(sentences):
    return [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences if sentence]

def match_sentences_with_keywords(sentences, keywords):
    return [(keyword, sentence) for sentence in sentences for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE)]

def store_matched_sentences_in_db(syllabus_id, matched_sentences):
    try:
        conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
        c = conn.cursor()
        for keyword, sentence in matched_sentences:
            c.execute("INSERT INTO syllabi (syllabus_id, keyword, sentence) VALUES (?, ?, ?)", (syllabus_id, keyword, sentence))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
def summarize(text):
    if not text.strip():
        return {"summary_text": "No content to summarize."}
    max_input_length = 1024
    if len(text) > max_input_length:
        text = text[:max_input_length]
    try:
        return summarizer(text, max_length=100, min_length=10, do_sample=False)
    except Exception as e:
        return {"summary_text": f"Error during summarization: {e}"}

def summarize_large_text(text):
    max_chunk_size = 1024
    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(summarize, text[i:i + max_chunk_size]) for i in range(0, len(text), max_chunk_size)]
        for future in concurrent.futures.as_completed(futures):
            try:
                summary = future.result()
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error during chunk summarization: {e}")
    return ' '.join(summaries)

def store_summary_in_db(keyword, summary):
    try:
        conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO summaries (keyword, summary) VALUES (?, ?)", (keyword, summary))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Text"])

def summarize_content(keyword):
    relevant_rows = df[df['Label'].str.contains(keyword, case=False, na=False)]
    all_summaries = relevant_rows['Summary'].tolist()
    
    # Convert all entries in all_summaries to strings, replacing non-strings with empty strings
    all_summaries = [str(summary) if not pd.isna(summary) else '' for summary in all_summaries]
    
    # If there are no summaries, use the text from the relevant rows
    if not all_summaries:
        all_texts = relevant_rows['Text'].tolist()
        all_summaries = [text for text in all_texts if text]
    
    summarized_text = ' '.join(all_summaries)
    return summarized_text

# Function to answer questions based on TF-IDF similarity
def answer_question(question):
    question_vec = tfidf_vectorizer.transform([question])
    cosine_similarities = cosine_similarity(question_vec, tfidf_matrix)
    similar_idx = cosine_similarities.argsort()[0][-1]
    return df.iloc[similar_idx]['Text']

@app.route('/answer', methods=['POST'])
def answer():
    data = request.json
    keyword = data.get('keyword', '')
    if keyword:
        answer = summarize_content(keyword)
        return jsonify({"answer": answer}), 200
    else:
        return jsonify({"error": "Keyword not provided"}), 400

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File is not a PDF"}), 400

    try:
        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({"error": "Failed to extract text from PDF"}), 500

        cleaned_text = clean_text(text)
        sentences = break_text_into_sentences(cleaned_text)
        formatted_sentences = format_sentences(sentences)
        matched_sentences = match_sentences_with_keywords(formatted_sentences, keywords)

        combined_text = ' '.join(sentence for _, sentence in matched_sentences)
        combined_summary = summarize_large_text(combined_text)

        # Store combined summary in the database
        conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO pdf_summaries (file_name, summary) VALUES (?, ?)", (file.filename, combined_summary))
        conn.commit()
        conn.close()

        # Prepare summaries by keyword
        summaries = {kw: summarize_large_text(' '.join(sentence for k, sentence in matched_sentences if k.lower() == kw)) for kw in keywords}
        
        return jsonify({"message": "File uploaded and processed", "file_summary": combined_summary, "summaries": summaries}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        text = extract_text_from_pdf(file)
        cleaned_text = clean_text(text)
        sentences = break_text_into_sentences(cleaned_text)
        formatted_sentences = format_sentences(sentences)
        matched_sentences = match_sentences_with_keywords(formatted_sentences, keywords)

        found_keywords = list(set(keyword for keyword, _ in matched_sentences))

        return jsonify({'keywords': found_keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_keyword', methods=['POST'])
def search_keyword():
    data = request.json
    keyword = data.get('keyword')
    print(f"Received keyword: {keyword}")  # Debugging
    
    if not keyword:
        return jsonify({"error": "Keyword not provided"}), 400

    conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
    c = conn.cursor()
    c.execute("SELECT summary FROM keyword_summaries WHERE keyword LIKE ?", ('%' + keyword + '%',))
    result = c.fetchall()
    conn.close()
    
    summaries = [row[0] for row in result]
    
    if not summaries:
        return jsonify({"error": f"No summaries found for keyword: {keyword}"}), 404

    print(f"Found summaries: {summaries}")  # Debugging
    return jsonify({"keyword": keyword, "summaries": summaries})

@app.route('/compare_summaries', methods=['POST'])
def compare_summaries():
    file = request.files.get('file')
    keyword = request.form.get('keyword', '').lower()

    if not file or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File is not a PDF"}), 400

    if not keyword:
        return jsonify({"error": "Keyword not provided"}), 400

    try:
        # Ensure 'extract_text_from_pdf' and other functions are implemented correctly
        text = extract_text_from_pdf(file)
        cleaned_text = clean_text(text)
        sentences = break_text_into_sentences(cleaned_text)
        formatted_sentences = format_sentences(sentences)
        
        # Log intermediate steps
        print(f"Extracted text: {text}")
        print(f"Cleaned text: {cleaned_text}")
        print(f"Sentences: {sentences}")
        print(f"Formatted sentences: {formatted_sentences}")

        matched_sentences = match_sentences_with_keywords(formatted_sentences, [keyword])
        found_keywords = list(set(kw for kw, _ in matched_sentences))
        
        # Log matched sentences and found keywords
        print(f"Matched sentences: {matched_sentences}")
        print(f"Found keywords: {found_keywords}")

        if keyword not in found_keywords:
            return jsonify({"error": f"Keyword '{keyword}' not found in the uploaded syllabus"}), 404

        combined_text = ' '.join(sentence for kw, sentence in matched_sentences if kw.lower() == keyword)
        user_summary = summarize_large_text(combined_text)

        conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
        c = conn.cursor()
        c.execute("SELECT summary FROM keyword_summaries WHERE keyword = ?", (keyword,))
        result = c.fetchone()
        model_summary = result[0] if result else "No precomputed summary available."
        conn.close()

        return jsonify({"user_summary": user_summary, "model_summary": model_summary, "keywords": found_keywords}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
