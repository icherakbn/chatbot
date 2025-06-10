import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="mysecret",
    host="rag-postgres",   # ✅ FIXED
    port="5432"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cursor.close()
conn.close()

print("✅ Table 'chat_history' created.")
