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
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)
CORS(app)
# Initialize summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = None

# Define keywords
keywords = ['Instructor', 'Email', 'Office', 'Late Work', 'Mentor', 'Course Description', 'Objective', 'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 'Attendance', 'Academic Integrity', 'Peer Mentor', 'Technology']

def init_db():
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS syllabi
                 (id INTEGER PRIMARY KEY, syllabus_id INTEGER, keyword TEXT, sentence TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS summaries
                 (id INTEGER PRIMARY KEY, keyword TEXT, summary TEXT)''')
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
    state_dict = torch.load('/u/irist_guest/Desktop/x/modelll.pth', map_location=torch.device('cpu'))
    num_labels = state_dict['output.weight'].size(0)
    model = LabelModel(num_labels)
    model.load_state_dict(state_dict)
    model.eval()

def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def check_similarity(new_summary, keyword):
    """
    Check if a new summary is similar to any existing summaries for the given keyword.
    """
    conn = sqlite3.connect('syllabi.db')
    c = conn.cursor()
    c.execute("SELECT summary FROM summaries WHERE keyword = ?", (keyword,))
    existing_summaries = [row[0] for row in c.fetchall()]
    conn.close()

    if not existing_summaries:
        return False

    # Compute similarity using SentenceTransformer embeddings
    embeddings = embedding_model.encode([new_summary] + existing_summaries)
    similarity_matrix = cosine_similarity([embeddings[0]], embeddings[1:])
    max_similarity = similarity_matrix.max()

    return max_similarity > 0.8  # Similarity threshold

def store_summary(keyword, summary):
    """
    Store the summary in the database if it's not similar to existing ones.
    """
    if not check_similarity(summary, keyword):
        conn = sqlite3.connect('syllabi.db')
        c = conn.cursor()
        c.execute("INSERT INTO summaries (keyword, summary) VALUES (?, ?)", (keyword, summary))
        conn.commit()
        conn.close()



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

def match_sentences_with_keywords(sentences, keywords):
    matched_sentences = []
    for sentence in sentences:
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE):
                matched_sentences.append((keyword, sentence))
                break
    return matched_sentences

# Store matched sentences in database
def store_matched_sentences_in_db(syllabus_id, matched_sentences):
    try:
        conn = sqlite3.connect('syllabi.db')
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
    for i in range(0, len(text), max_chunk_size):
        chunk = text[i:i + max_chunk_size]
        summary = summarize(chunk)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)


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

        with sqlite3.connect('syllabi.db') as conn:
            c = conn.cursor()
            c.execute("SELECT MAX(syllabus_id) FROM syllabi")
            result = c.fetchone()
            new_syllabus_id = result[0] + 1 if result[0] is not None else 1

        store_matched_sentences_in_db(new_syllabus_id, matched_sentences)
        return jsonify({"message": "File uploaded and processed", "syllabus_id": new_syllabus_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

        return jsonify({"error": str(e)}), 500

@app.route('/compare_summaries', methods=['POST'])
def compare_summaries():
    keyword = request.form.get('keyword', '').lower()
    file = request.files.get('file')

    if not keyword or not file:
        return jsonify({"error": "Keyword or file missing"}), 400

    try:
        df = pd.read_csv('/u/irist_guest/Desktop/x/cw.csv')
        relevant_sentences = df[df['Text'].str.contains(keyword, case=False, na=False)]
        combined_text_model = ' '.join(relevant_sentences['Text'].tolist())
        summary_model = summarize_large_text(combined_text_model)

        full_text_pdf = extract_text_from_pdf(file)
        sentences_pdf = [sentence for sentence in break_text_into_sentences(full_text_pdf) if keyword in sentence.lower()]
        combined_text_pdf = ' '.join(sentences_pdf)
        summary_pdf = summarize_large_text(combined_text_pdf)

        return jsonify({
            'model_summary': summary_model,
            'pdf_summary': summary_pdf
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Summarize entire syllabus endpoint
import spacy
from spacy.lang.en import English

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def extract_key_sentences(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    # Simple sentence ranking based on length; you can refine this
    ranked_sentences = sorted(sentences, key=lambda sent: len(sent), reverse=True)
    # Select top sentences for summarization
    top_sentences = ranked_sentences[:5]  # Adjust number as needed
    return ' '.join([sent.text for sent in top_sentences])

@app.route('/summarize_entire_syllabus', methods=['POST'])
def summarize_entire_syllabus():
    file = request.files.get('file')

    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        full_text = extract_text_from_pdf(file)
        sentences = break_text_into_sentences(full_text)
        matched_sentences = match_sentences_with_keywords(sentences, keywords)
        combined_text = ' '.join([sentence for _, sentence in matched_sentences])

        key_sentences_text = extract_key_sentences(combined_text)
        summary = summarize_large_text(key_sentences_text)

        return jsonify({'summary': summary}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def split_text(text, max_length):
    """Split text into chunks of max_length tokens."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word)
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

import logging

logging.basicConfig(level=logging.DEBUG)

@app.route('/search_keyword', methods=['POST'])
def search_keyword():
    data = request.json
    keyword = data.get('keyword')

    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    try:
        logging.debug(f"Received keyword: {keyword}")

        df = pd.read_csv('/u/irist_guest/Desktop/x/cw.csv')
        keyword_lower = keyword.lower()
        
        matched_sentences = []

        for index, row in df.iterrows():
            text = row['Text']
            if keyword_lower in text.lower():
                matched_sentences.append(text)
        
        logging.debug(f"Matched sentences count: {len(matched_sentences)}")

        if not matched_sentences:
            return jsonify({"summaries": []}), 200

        combined_text = ' '.join(matched_sentences)
        chunks = split_text(combined_text, 500)
        summaries = [summarizer(chunk, max_length=75, min_length=10, do_sample=False)[0]['summary_text'] for chunk in chunks]

        logging.debug(f"Summaries generated: {summaries}")

        return jsonify({"summaries": summaries}), 200

    except Exception as e:
        logging.error(f"Error in /search_keyword: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route('/search_keyword', methods=['POST'])
# def search_keyword():
#     data = request.json
#     keyword = data.get('keyword')

#     if not keyword:
#         return jsonify({"error": "No keyword provided"}), 400

#     try:
#         df = pd.read_csv('/u/irist_guest/Desktop/x/cw.csv')
#         keyword_lower = keyword.lower()
        
#         matched_sentences = []

#         for index, row in df.iterrows():
#             text = row['Text']
#             if keyword_lower in text.lower():
#                 matched_sentences.append(text)
        
#         if not matched_sentences:
#             return jsonify({"summaries": []}), 200

#         combined_text = ' '.join(matched_sentences)
#         chunks = split_text(combined_text, 500)
#         summaries = [summarizer(chunk, max_length=75, min_length=10, do_sample=False)[0]['summary_text'] for chunk in chunks]

#         # Store summaries in the database
#         for summary in summaries:
#             store_summary(keyword, summary)

#         return jsonify({"summaries": summaries}), 200

#     except Exception as e:
#         print(f"Error in /search_keyword: {e}")
#         print(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500



@app.route('/')
def welcome():
    return "Welcome to the Syllabus Q&A API!"

if __name__ == '__main__':
    init_db()
    init_model()
    app.run(debug=True)



# @app.route('/search_keyword', methods=['POST'])
# def search_keyword():
#     data = request.json
#     keyword = data.get('keyword')

#     if not keyword:
#         return jsonify({"error": "No keyword provided"}), 400

#     try:
#         df = pd.read_csv('/u/irist_guest/Desktop/x/cw.csv')
#         keyword_lower = keyword.lower()
        
#         matched_sentences = []

#         for index, row in df.iterrows():
#             text = row['Text']
#             if keyword_lower in text.lower():
#                 matched_sentences.append(text)
        
#         if not matched_sentences:
#             return jsonify({"instances": []}), 200

#         # Compute embeddings
#         embeddings = embedding_model.encode(matched_sentences, convert_to_tensor=True)
#         embeddings = embeddings.cpu().detach().numpy()  # Convert to numpy array

#         # Compute cosine similarity
#         cosine_scores = cosine_similarity(embeddings)

#         unique_sentences = []
#         used_indices = set()

#         for i, sentence in enumerate(matched_sentences):
#             if i in used_indices:
#                 continue
#             unique_sentences.append(sentence)
#             for j, score in enumerate(cosine_scores[i]):
#                 if i != j and score > 0.8:  # Threshold for similarity
#                     used_indices.add(j)

#         return jsonify({"instances": unique_sentences}), 200

#     except Exception as e:
#         print(f"Error in /search_keyword: {e}")
#         print(traceback.format_exc())