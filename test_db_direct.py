import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=9999,
    user="super",
    password="Lx9bR4v0P@s55!",
    dbname="jbravo",
    connect_timeout=15,
)

cur = conn.cursor()
cur.execute("select now()")
print("DB time:", cur.fetchone())

cur.execute("select current_database(), current_user")
print("DB info:", cur.fetchone())

conn.close()
