import os 
import PyPDF2
pdf_dir = os.getcwd()
for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_file = open(os.path.join(pdf_dir, filename), 'rb')
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = '""'
            for i in range (len(pdf_reader.pages)):
                page = pdf_reader.pages[i]
                text += page.extract_text()
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_file = open(os.path.join(pdf_dir, txt_filename), 'w')
            txt_file.write(text)
            pdf_file.close()
            txt_file.close()
