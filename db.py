import sqlite3

conn = sqlite3.connect('spam_filter_table.db')
print("Opened database successfully")

conn.execute('CREATE TABLE email (id INTEGER PRIMARY KEY AUTOINCREMENT , BODY TEXT, status BIT )')
print("Table created successfully")
conn.close()