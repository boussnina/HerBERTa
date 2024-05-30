import os
import requests
import pandas as pd
import time
import random

IMAGE_PATH = "../HERBARIUM_SHEETS"
DOWNLOAD_PATH = "../greenland_herb.csv"

if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

class ImageDownloader:

    def __init__(self, download_path):
        self.df = pd.read_csv(download_path).dropna(subset=['1,111-collectionobjectattachments,41.attachment.attachmentLocation'])
        self.df.rename(columns={'1,111-collectionobjectattachments,41.attachment.attachmentLocation': 'image'}, inplace=True)

    def download_images_sequential(self, amount_of_images=10):
        downloaded_count = 0
        for image_url in self.df['image']:
            if downloaded_count >= amount_of_images:
                break
            
            if image_url.startswith("NHMD"):
                name = image_url.split("/")[-1].split(".")[0]
                url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.jpg"
            else:
                name = image_url.split("/")[-1].split(".")[0]
                url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.att.jpg"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                file_path = os.path.join(IMAGE_PATH, f"{name}.jpg")
            
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            
                print(f"Successfully downloaded {name}.jpg")
                downloaded_count += 1
            
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {name}.jpg: {e}")
            
    def download_random_images(self, amount_of_random_images=1):
        random_images = self._pick_random_images(amount_of_random_images)
        for image_url in random_images:
            if image_url.startswith("NHMD"):
                name = image_url.split("/")[-1].split(".")[0]
                url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.jpg"
            else:
                name = image_url.split("/")[-1].split(".")[0]
                url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.att.jpg"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                file_path = os.path.join(IMAGE_PATH, f"{name}.jpg")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {name}.jpg")
            
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {name}.jpg: {e}")

        
    def _pick_random_images(self, amount_of_random_images=1):
        return self.df.sample(n=amount_of_random_images)['image'].values
        

if __name__ == '__main__':
    downloader = ImageDownloader(DOWNLOAD_PATH)
    downloader.download_random_images(3)  
