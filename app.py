from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import fitz  # PyMuPDF
import os
from transformers import pipeline

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(pdf_path)
            txt_path = convert_pdf_to_txt(pdf_path)
            flash('File successfully uploaded and converted')
            return send_file(txt_path, as_attachment=True)
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(request.url)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        question = request.form['question']
        txt_file = request.form['txt_file']
        if txt_file and question:
            answer = answer_question(txt_file, question)
            return {'question': question, 'answer': answer}
        return {'error': 'Missing text file or question'}
    except Exception as e:
        return {'error': f'An error occurred: {str(e)}'}

def convert_pdf_to_txt(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    txt_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(pdf_path).replace('.pdf', '.txt'))
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)
    return txt_path

def answer_question(txt_path, question):
    with open(txt_path, 'r', encoding='utf-8') as file:
        context = file.read()

    # Load a pre-trained language model
    model_name = 'distilbert-base-uncased-distilled-squad'
    qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

    # Get the answer to the question
    result = qa_pipeline({'context': context, 'question': question})
    return result['answer']

if __name__ == '__main__':
    app.run(debug=True, port=5001)
