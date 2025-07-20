import sqlite3

DB_FILE = "detections.db"  # Ensure this is in your app's working directory

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        total_trees INTEGER,
        young_coffee INTEGER,
        mature_coffee INTEGER,
        dead_coffee INTEGER,
        avg_spacing REAL,
        smallest_spacing REAL,
        largest_spacing REAL,
        annotated_path TEXT,
        pdf_path TEXT,
        geojson_path TEXT,
        csv_path TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def fetch_all_detections():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM detections ORDER BY id DESC")
    records = c.fetchall()
    conn.close()
    return records

def insert_detection(filename, total, young, mature, dead, avg, small, large, annot_path, pdf_path, geo_path, csv_path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO detections (
        filename, total_trees, young_coffee, mature_coffee, dead_coffee,
        avg_spacing, smallest_spacing, largest_spacing,
        annotated_path, pdf_path, geojson_path, csv_path,
        timestamp
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        filename, total, young, mature, dead, avg, small, large,
        annot_path, pdf_path, geo_path, csv_path
    ))
    conn.commit()
    conn.close()

def delete_detection(record_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
