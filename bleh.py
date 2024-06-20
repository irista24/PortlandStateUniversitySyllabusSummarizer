import pandas as pd
import os
file = '/u/irist_guest/Desktop/csv/a.csv'
print("file defined")
df = pd.read_csv(file, encoding = 'latin-1', on_bad_lines = 'skip')
print("file read")
keywords = ['Instructor', 'Email', 'Office Hours', 'Late Work', 'Peer Mentor', 'Course Description', 'Learning Objective']  

for keyword in keywords:
    df[keyword] = ''
    print("keyword function working")

current_keyword = None
for index, row in df.iterrows():
    print("iterrating through the columns working")
    text = row
    
    if any(text.startswith(keyword)):
        for keyword in keywords:
            current_keyword = next((keyword for keyword in keywords if text.startswith(keyword)), None)
            print("changing keywords working")
            continue

    if current_keyword:
        df.at[index, current_keyword] = row['Text']
        print("if keyword is true working")


collapsed_data = {keyword: ' '.join(df[keyword].values).strip() for keyword in keywords}
collapsed_df = pd.DataFrame([collapsed_data])

output_file_path = '/u/irist_guest/newfile.csv'
collapsed_df.to_csv(output_file_path, index=False)

print(collapsed_df.head())
