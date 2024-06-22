import pandas as pd

file = '/u/irist_guest/Desktop/pdfs/yu.csv'

# Read the CSV file
df = pd.read_csv(file,encoding = 'utf-8')

# Define keywords
keywords = ['Instructor', 'Email', 'Office', 'Late', 'Mentor', 'Description', 'Objective']

# Initialize columns for each keyword
for keyword in keywords:
    df[keyword] = ''

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    text = str(row['Header'])  # Ensure 'Header' column is correctly referenced and converted to string if necessary
    
    for keyword in keywords:
        if text.startswith(keyword):
            # Capture the text following the keyword
            text_after_keyword = text[len(keyword):].strip()
            
            # Assign the captured text to the corresponding column
            df.at[index, keyword] = text_after_keyword
            break

# Output the updated dataframe to a new CSV file
output_file_path = '/u/irist_guest/results.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # utf-8-sig for handling special characters properly

# Print the first few rows of the updated dataframe (optional)
print(df.head())
