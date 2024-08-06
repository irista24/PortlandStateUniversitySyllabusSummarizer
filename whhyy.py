from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

@app.route('/search_keyword', methods=['POST'])
def search_keyword():
    data = request.json
    keyword = data.get('keyword', '').strip()
    print(f'Received keyword: {keyword}')

    conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
    cursor = conn.cursor()

    # Log the content of the summaries table
    cursor.execute('SELECT * FROM summaries')
    all_summaries = cursor.fetchall()
    print(f'All summaries in DB: {all_summaries}')

    cursor.execute('SELECT summary FROM summaries WHERE keyword = ? COLLATE NOCASE', (keyword,))
    results = cursor.fetchall()
    conn.close()

    summaries = [result[0] for result in results]
    print(f'Query results: {summaries}')

    return jsonify({'summaries': summaries})

if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
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
# from transformers import pipeline
# from sklearn.metrics.pairwise import cosine_similarity
# import traceback
# from sentence_transformers import SentenceTransformer
# import concurrent.futures

# app = Flask(__name__)
# CORS(app)

# # Initialize summarizer and embedding model
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Define keywords
# keywords = [
#     'Instructor', 'Email', 'Office', 'Late Work','Course Description', 'Objective', 
#     'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 
#     'Attendance', 'Academic Integrity', 'Technology'
# ]

# def init_db():
#     conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS syllabi
#                  (id INTEGER PRIMARY KEY, syllabus_id INTEGER, keyword TEXT, sentence TEXT)''')
#     c.execute('''CREATE TABLE IF NOT EXISTS summaries
#                  (id INTEGER PRIMARY KEY, keyword TEXT, summary TEXT)''')
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
#     state_dict = torch.load('/u/irist_guest/syllabus-app/x/modelll.pth', map_location=torch.device('cpu'))
#     num_labels = state_dict['output.weight'].size(0)
#     model = LabelModel(num_labels)
#     model.load_state_dict(state_dict)
#     model.eval()

# def extract_text_from_pdf(file):
#     try:
#         with pdfplumber.open(io.BytesIO(file.read())) as pdf:
#             text = ""
#             for page in pdf.pages:
#                 text += page.extract_text() or ""
#             return text
#     except Exception as e:
#         return f"Error extracting text from PDF: {e}"

# def check_similarity(new_summary, keyword):
#     conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#     c = conn.cursor()
#     c.execute("SELECT summary FROM summaries WHERE keyword = ?", (keyword,))
#     existing_summaries = [row[0] for row in c.fetchall()]
#     conn.close()

#     if not existing_summaries:
#         return False

#     embeddings = embedding_model.encode([new_summary] + existing_summaries)
#     similarity_matrix = cosine_similarity([embeddings[0]], embeddings[1:])
#     max_similarity = similarity_matrix.max()

#     return max_similarity > 0.8

# def store_summary(keyword, summary):
#     if not check_similarity(summary, keyword):
#         conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO summaries (keyword, summary) VALUES (?, ?)", (keyword, summary))
#         conn.commit()
#         conn.close()

# def clean_text(text):
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = re.sub(r'\n+', ' ', text)
#     text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
#     return text

# def is_valid_sentence(sentence):
#     return len(sentence.strip()) > 0

# def break_text_into_sentences(text):
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     return sentences

# def format_sentences(sentences):
#     formatted_sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences if sentence]
#     return formatted_sentences

# def match_sentences_with_keywords(sentences, keywords):
#     matched_sentences = [(keyword, sentence) for sentence in sentences for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', sentence, re.IGNORECASE)]
#     return matched_sentences

# def store_matched_sentences_in_db(syllabus_id, matched_sentences):
#     try:
#         conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#         c = conn.cursor()
#         for keyword, sentence in matched_sentences:
#             c.execute("INSERT INTO syllabi (syllabus_id, keyword, sentence) VALUES (?, ?, ?)", (syllabus_id, keyword, sentence))
#         conn.commit()
#     except sqlite3.Error as e:
#         print(f"Database error: {e}")
#     finally:
#         conn.close()

# def summarize(text):
#     if not text.strip():
#         return {"summary_text": "No content to summarize."}
#     max_input_length = 1024
#     if len(text) > max_input_length:
#         text = text[:max_input_length]
#     try:
#         return summarizer(text, max_length=100, min_length=10, do_sample=False)
#     except Exception as e:
#         return {"summary_text": f"Error during summarization: {e}"}

# def summarize_large_text(text):
#     max_chunk_size = 1024
#     summaries = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(summarize, text[i:i + max_chunk_size]) for i in range(0, len(text), max_chunk_size)]
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 summary = future.result()
#                 summaries.append(summary[0]['summary_text'])
#             except Exception as e:
#                 print(f"Error during chunk summarization: {e}")
#     return ' '.join(summaries)


# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if not file.filename.lower().endswith('.pdf'):
#         return jsonify({"error": "File is not a PDF"}), 400

#     try:
#         text = extract_text_from_pdf(file)
#         if not text:
#             return jsonify({"error": "Failed to extract text from PDF"}), 500

#         cleaned_text = clean_text(text)
#         sentences = break_text_into_sentences(cleaned_text)
#         formatted_sentences = format_sentences(sentences)
#         matched_sentences = match_sentences_with_keywords(formatted_sentences, keywords)

#         with sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db') as conn:
#             c = conn.cursor()
#             c.execute("SELECT MAX(syllabus_id) FROM syllabi")
#             result = c.fetchone()
#             new_syllabus_id = result[0] + 1 if result[0] is not None else 1

#         store_matched_sentences_in_db(new_syllabus_id, matched_sentences)
#         return jsonify({"message": "File uploaded and processed", "syllabus_id": new_syllabus_id}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/compare_summaries', methods=['POST'])
# def compare_summaries():
#     keyword = request.form.get('keyword', '').lower()
#     file = request.files.get('file')

#     if not keyword or not file:
#         return jsonify({"error": "Keyword and file are required"}), 400

#     if not file.filename.lower().endswith('.pdf'):
#         return jsonify({"error": "File is not a PDF"}), 400

#     try:
#         text = extract_text_from_pdf(file)
#         cleaned_text = clean_text(text)
#         sentences = break_text_into_sentences(cleaned_text)
#         formatted_sentences = format_sentences(sentences)
#         matched_sentences = [(kw, sentence) for kw, sentence in match_sentences_with_keywords(formatted_sentences, keywords) if kw.lower() == keyword]

#         if not matched_sentences:
#             return jsonify({"error": f"No sentences matched for keyword: {keyword}"}), 404

#         combined_text = ' '.join(sentence for _, sentence in matched_sentences)
#         user_summary = summarize_large_text(combined_text)

#         conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#         c = conn.cursor()
#         c.execute("SELECT summary FROM summaries WHERE keyword = ?", (keyword,))
#         result = c.fetchone()
#         model_summary = result[0] if result else "No precomputed summary available."

#         return jsonify({"user_summary": user_summary, "model_summary": model_summary}), 200

#     except Exception as e:
#         traceback.print_exc()
        
# @app.route('/search_keyword', methods=['POST'])
# def search_keyword():
#     data = request.json
#     keyword = data.get('keyword', '').strip().upper()
#     print(f'Received keyword: {keyword}')  # Debugging line

#     # Connect to your database
#     conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#     cursor = conn.cursor()

#     # Query the database
#     cursor.execute('SELECT summary FROM summaries WHERE keyword = ?', (keyword,))
#     results = cursor.fetchall()
#     conn.close()

#     summaries = [result[0] for result in results]
#     print(f'Query results: {summaries}')  # Debugging line

#     return jsonify({"summaries": summaries}), 200





# @app.route('/')
# def welcome():
#      return jsonify({"message": "Welcome to the Syllabus Summarization API!"})

# if __name__ == "__main__":
#     init_db()
#     init_model()
#     app.run(host='127.0.0.1', port=5000, debug=True)

