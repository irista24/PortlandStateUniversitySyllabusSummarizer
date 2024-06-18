import os  
import fnmatch  
import pandas as pd  

files_dir = r'/u/irist_guest/Desktop/pdfs/'

# Check if directory exists
if not os.path.exists(files_dir):
    print(f"Directory does not exist: {files_dir}")
else:
    files = os.listdir(files_dir)

    print(f"Files in directory: {files}")

    for file in files:
        print(f"Checking file: {file}")
        if fnmatch.fnmatch(file, 'pdfs*'):
            print(f"File matches pattern: {file}")
            extension = os.path.splitext(file)[1]
            if extension == '.txt':
                print(f"File has .txt extension: {file}")
                filename = os.path.join(files_dir, file)
                
                # Debug output to verify file processing
                print(f"Processing file: {filename}")
                
                try:
                    # Reading the text file assuming '|' as a delimiter
                    df = pd.read_csv(filename, sep='|')
                    
                    # Generating new filename for CSV
                    new_filename = os.path.splitext(filename)[0] + '.csv'
                    
                    # Saving the DataFrame to CSV
                    df.to_csv(new_filename, index=False)
                    
                    # Debug output to confirm successful write
                    print(f"Successfully converted {filename} to {new_filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
            else:
                print(f"File does not have .txt extension: {file}")
        else:
            print(f"File does not match pattern: {file}")