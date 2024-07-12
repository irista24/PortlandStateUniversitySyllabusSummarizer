import sqlite3

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect('syllabi.db')

# Create a table if it doesn't exist
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS syllabi
             (id INTEGER PRIMARY KEY, label TEXT, text TEXT)''')

# Commit changes and close connection
conn.commit()
conn.close()

print("SQLite database and table created successfully!")
