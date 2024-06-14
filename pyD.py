from PyPDF2 import PdfReader
import glob
def extract_text_from_pdf(pdf_path):
    
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_number in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_number]
            text = page.extract_text()
    return text

pdf_files = glob.glob('/u/irist_guest/Desktop/pdfs/*.pdf')

extracted_texts = []
for pdf_file in pdf_files:
    text = extract_text_from_pdf(pdf_file)
    extracted_texts.append(text)

print(extracted_texts)