from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load the CSV data
df = pd.read_csv('/u/irist_guest/Desktop/pdfs/aw.csv')

# Clean the text in the dataframe
df["Text"] = df['Text'].fillna('').apply(lambda x: x.replace("\n", "").strip())

# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Text"])

# Function to answer questions
def answer_question(question):
    question_vec = tfidf_vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vec, tfidf_matrix)
    most_similar_index = similarity_scores.argmax()
    most_similar_text = df.iloc[most_similar_index]["Text"]
    return most_similar_text

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    answer = answer_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
