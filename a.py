
# import sqlite3

# def initialize_database():
#     conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#     c = conn.cursor()
    
#     # Create table if it doesn't exist
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS summaries (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             keyword TEXT NOT NULL,
#             summary TEXT NOT NULL
#         )
#     ''')
    
#     conn.commit()
#     conn.close()
# import sqlite3
# import pandas as pd
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# import concurrent.futures

# # Initialize summarizer and embedding model
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Define keywords
# keywords = [
#     'Instructor', 'Email', 'Office', 'Late Work', 'Course Description', 'Objective', 
#     'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 
#     'Attendance', 'Academic Integrity', 'Technology'
# ]

# def summarize(text):
#     if not text.strip():
#         return {"summary_text": "No content to summarize."}
#     max_input_length = 1024
#     if len(text) > max_input_length:
#         text = text[:max_input_length]
#     try:
#         return summarizer(text, max_length=50, min_length=5, do_sample=False)
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

# def precompute_summaries():
#     # Initialize database
#     initialize_database()

#     df = pd.read_csv('/u/irist_guest/syllabus-app/x/cw.csv')
#     keyword_summaries = {keyword: [] for keyword in keywords}
    
#     for index, row in df.iterrows():
#         text = row['Text']
#         for keyword in keywords:
#             if keyword.lower() in text.lower():
#                 keyword_summaries[keyword].append(text)
    
#     conn = sqlite3.connect('/u/irist_guest/syllabus-app/x/syllabi.db')
#     c = conn.cursor()

#     summaries_to_insert = []

#     def summarize_text_chunk(text_chunk):
#         return summarize_large_text(text_chunk)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for keyword, sentences in keyword_summaries.items():
#             if sentences:
#                 combined_text = ' '.join(sentences)
#                 futures.append(executor.submit(summarize_text_chunk, combined_text))
        
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 keyword = list(keyword_summaries.keys())[futures.index(future)]
#                 summary = future.result()
#                 print(f"Summary for {keyword}: {summary}")  # Print to verify summary
#                 summaries_to_insert.append((keyword, summary))
#             except Exception as e:
#                 print(f"Error during summarization: {e}")

#     # Print to verify summaries_to_insert
#     print("Summaries to insert:")
#     for item in summaries_to_insert:
#         print(item)

#     if summaries_to_insert:
#         c.executemany("INSERT INTO summaries (keyword, summary) VALUES (?, ?)", summaries_to_insert)
#         conn.commit()
    
#     conn.close()

# if __name__ == "__main__":
#     precompute_summaries()


