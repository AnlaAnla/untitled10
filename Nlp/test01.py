from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import sqlite3
import json
import base64

with open(r"C:\Code\ML\Text\test\Local State", 'r', encoding='utf-8') as f:
    local_state = json.loads(f.read())
    # master_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])
    # master_key = master_key[5:]

print(key)

conn = sqlite3.connect(r"C:\Code\ML\Text\test\Login Data")
cursor = conn.cursor()

cursor.execute('SELECT action_url, username_value, password_value FROM logins')

rows = cursor.fetchall()

for row in rows:
    print(row)


cursor.close()
conn.close()

print()