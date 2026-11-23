import tomllib
import psycopg2

# Read database credentials from secrets.toml
with open(".streamlit/secrets.toml", "rb") as file:
    config = tomllib.load(file)

db_config = config["postgres"]

try:
    connection = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        dbname=db_config["database"],
        user=db_config["user"],
        password=db_config["password"]
    )

    print("PostgreSQL connection successful!")

    cursor = connection.cursor()
    cursor.execute("SELECT current_database();")
    print("Connected database:", cursor.fetchone()[0])

    cursor.close()
    connection.close()

except Exception as error:
    print("Database connection failed:")
    print(error)