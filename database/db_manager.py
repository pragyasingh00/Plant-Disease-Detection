import psycopg2
import streamlit as st


def get_connection():
    """Create and return a PostgreSQL connection."""
    config = st.secrets["postgres"]

    return psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["database"],
        user=config["user"],
        password=config["password"]
    )


def save_prediction(image_name, predicted_class, confidence):
    """Save one prediction to the database."""
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediction_history
                (image_name, predicted_class, confidence)
            VALUES (%s, %s, %s);
            """,
            (image_name, predicted_class, confidence)
        )

        connection.commit()
        cursor.close()

    finally:
        connection.close()


def get_prediction_history():
    """Retrieve all saved predictions, newest first."""
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, image_name, predicted_class, confidence, created_at
            FROM prediction_history
            ORDER BY created_at DESC;
            """
        )

        records = cursor.fetchall()
        cursor.close()

        return records

    finally:
        connection.close()