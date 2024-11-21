import psycopg2
from datetime import datetime

def upload_image(image_id, filename, filepath, caption):
    try:
        # Pull this out into a function
        # And read in credentials from env
        conn = psycopg2.connect( # TODO: Read in credentials from env
            dbname="user_logs",
            user="logger",
            password="secure_password",
            host="postgres"
        )

        user_id = 1
        ip_address = "127.0.0.1"
        timestamp = datetime.now()
        
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
