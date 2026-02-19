import os
import psycopg2

print("DATABASE_URL present:", bool(os.getenv("DATABASE_URL")))

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("select now()")
print("DB time:", cur.fetchone())
conn.close()
