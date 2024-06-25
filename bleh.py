import pandas as pd

# Read the CSV file
file = '/u/irist_guest/Desktop/pdfs/mu.csv'
df = pd.read_csv(file, encoding='utf-8')

# Define keywords
keywords = ['Instructor', 'Email', 'Office', 'Late Work', 'Mentor', 'Course Description', 'Objective', 'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 'Attendance', 'Academic Integrity', 'Peer Mentor', 'Technology']

# Initialize a DataFrame with the keyword columns
df_result = pd.DataFrame(columns=keywords)

# Initialize variables to track current keyword and associated data
current_keyword = None
current_data = []

# Function to process and store data for each keyword
def process_data_for_keyword(keyword, data):
    if data:
        data_string = " ".join(data)
        df_result.loc[len(df_result), keyword] = data_string

# Iterate through each row to capture data based on keywords
for index, row in df.iterrows():
    header_text = str(row['Header']).strip()
    
    # Check if any keyword is in the current header text
    keyword_found = False
    for keyword in keywords:
        if keyword.lower() in header_text.lower():
            if current_keyword is not None:
                # Process and store data for the previous keyword
                process_data_for_keyword(current_keyword, current_data)
                current_data = []  # Reset current_data
            
            # Update current_keyword
            current_keyword = keyword
            keyword_found = True
            break
    
    # If a keyword is found or already processing, collect the header text
    if keyword_found or current_keyword is not None:
        current_data.append(header_text)

# Process and store data for the last keyword block
if current_keyword:
    process_data_for_keyword(current_keyword, current_data)

# Output the updated DataFrame to a new CSV file
output_file_path = '/u/irist_guest/Desktop/pdfs/wy.csv'
df_result.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# Print or further process the consolidated DataFrame as needed
print(f"Consolidated data saved to {output_file_path}")
print(df_result.head())


#  import pandas as pd

# # Read the CSV file
# file = '/u/irist_guest/Desktop/pdfs/mu.csv'
# df = pd.read_csv(file, encoding='utf-8')

# # Define keywords
# keywords = ['Instructor', 'Email', 'Office', 'Late Work', 'Mentor', 'Course Description', 'Objective', 'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 'Resources', 'Attendance', 'Academic Integrity', 'Peer Mentor', 'Technology']

# # Initialize an empty DataFrame to store data for each keyword
# keyword_data = {keyword: [] for keyword in keywords}

# # Initialize variables to track current keyword and associated data
# current_keyword = None
# current_data = []

# # Function to process and store data for each keyword
# def process_data_for_keyword(keyword, data):
#     if not data.empty:
#         df.loc[data.index, keyword] = data['Header'].astype(str).tolist()

# # Iterate through each row to capture data based on keywords
# for index, row in df.iterrows():
#     header_text = str(row['Header']).strip()
    
#     # Check if any keyword is in the current header text
#     keyword_found = False
#     for keyword in keywords:
#         if keyword.lower() in header_text.lower():
#             if current_keyword is not None:
#                 # Process and store data for the previous keyword
#                 process_data_for_keyword(current_keyword, pd.concat(current_data).reset_index(drop=True))
#                 current_data = []  # Reset current_data
            
#             # Update current_keyword
#             current_keyword = keyword
#             keyword_found = True
#             break
    
#     # If a keyword is found or already processing, collect row
#     if keyword_found or current_keyword is not None:
#         current_data.append(row.to_frame().T)

# # Process and store data for the last keyword block
# if current_keyword:
#     process_data_for_keyword(current_keyword, pd.concat(current_data).reset_index(drop=True))

# # Drop the original 'Header' column if no longer needed
# df.drop(columns=['Header'], inplace=True)

# # Output the updated dataframe to a new CSV file
# output_file_path = '/u/irist_guest/Desktop/pdfs/wy.csv'
# df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

# # Print or further process the consolidated DataFrame as needed
# print(f"Consolidated data saved to {output_file_path}")
# print(df.head())