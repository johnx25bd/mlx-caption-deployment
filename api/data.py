from datetime import datetime
import os
from pathlib import Path
import psycopg2

UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

async def save_file_to_disk(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    contents = await uploaded_file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return filename, file_path

def save_image_data_to_db(image_id, filename, filepath, caption):
    try:
        # TODO Pull this out into a function
        # And read in credentials from env
        conn = psycopg2.connect(
            dbname="user_logs",
            user="logger",
            password="secure_password",
            host="postgres"
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_images (
                image_id,
                caption,
                filename,
                filepath
            ) VALUES (
                %s,%s,%s,%s
            )
            """,
            (image_id, caption,filename, filepath)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to save image data: {str(e)}")