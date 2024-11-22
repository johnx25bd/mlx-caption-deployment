import psycopg2
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader

class PostgresImageCaptionDataset(Dataset):
    def __init__(self, images_dir="/app/images"):
        """
        Initialize dataset with path to images directory
        
        Args:
            images_dir (str): Path to the mounted images directory
        """
        self.images_dir = images_dir
        
        self.db_params = {
            'dbname': 'user_logs',
            'user': 'logger',
            'password': 'secure_password',
            'host': 'postgres',
            'port': 5432
        }
        
        # Connect and get all image data
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT image_id, caption, finetuned, filepath 
                    FROM user_images
                    WHERE finetuned = false
                    ORDER BY timestamp DESC
                """)
                self.data = cur.fetchall()
        print(f"data:{self.data}")
    
    def _get_connection(self):
        """Create and return a database connection"""
        return psycopg2.connect(**self.db_params)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset
        
        Args:
            idx (int): Index of the item to fetch
            
        Returns:
            dict: Dictionary containing:
                - image: Image loaded from filepath
                - caption: String caption
                - image_id: Unique identifier for the image
        """
        print(f"data: {self.data}")
        image_id, caption, finetuned, filepath = self.data[idx]
        
        full_path = os.path.join(self.images_dir, filepath)
        
        try:
            image = Image.open(full_path)
            
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return None
        
        return {
            'image': image,
            'caption': caption,
            'image_id': image_id
        }

if __name__ == "__main__":
    dataset = PostgresImageCaptionDataset()
    
    sample = dataset[0]
    if sample:
        print(f"Image ID: {sample['image_id']}")
        print(f"Caption: {sample['caption']}")
        print(f"Image size: {sample['image'].size}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)