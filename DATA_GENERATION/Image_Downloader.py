import os
import requests
import pandas as pd
import random


class ImageDownloader:

    def __init__(self, download_to="", download_from="greenland_herb.csv"):
        self.df = pd.read_csv(download_from).dropna(subset=['1,111-collectionobjectattachments,41.attachment.attachmentLocation'])
        self.df.rename(columns={'1,111-collectionobjectattachments,41.attachment.attachmentLocation': 'image'}, inplace=True)
        self.download_to = download_to or os.getcwd()
        if not os.path.exists(self.download_to):
            os.makedirs(self.download_to)

    def _download_image(self, image_url):
        if image_url.startswith("NHMD"):
            name = image_url.split("/")[-1].split(".")[0]
            url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.jpg"
        else:
            name = image_url.split("/")[-1].split(".")[0]
            url = f"https://specify-attachments.science.ku.dk/static/NHMD_Botany/originals/{name}.att.jpg"

        try:
            response = requests.get(url)
            response.raise_for_status()
            file_path = os.path.join(self.download_to, f"{name}.jpg")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {name}.jpg")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {name}.jpg: {e}")

    def download_images_sequential(self, amount_of_images=10):
        downloaded_count = 0
        for image_url in self.df['image']:
            if downloaded_count >= amount_of_images:
                break
            self._download_image(image_url)
            downloaded_count += 1

    def download_random_images(self, amount_of_random_images=1):
        random_images = self._pick_random_images(amount_of_random_images)
        for image_url in random_images:
            self._download_image(image_url)

    def _pick_random_images(self, amount_of_random_images=1):
        return self.df.sample(n=amount_of_random_images)['image'].values


if __name__ == '__main__':

    downloader = ImageDownloader()
    downloader._download_image("NHMD-688772.jpg")
    
    
    
    
