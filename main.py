import math
import os
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PIL import Image, ImageDraw
import pandas as pd
from thefuzz import fuzz
import re
import argparse
import spacy
from spacy import displacy
from ultralytics import YOLO
from htrocr.run import NHMDPipeline as pipe
from DATA_GENERATION.Image_Downloader import ImageDownloader 

# Load configuration
CONFIG_PATH = 'config.json'
with open(CONFIG_PATH, 'r') as f:
   config = json.load(f)

TEMP_DIR = "tempory_files"
IMAGE_DIR = "IMAGES"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Label Detector Component
class LabelDetector:
    """
    Class for detecting bounding boxes around labels in an image using the YOLOv8 model.
    """

    def __init__(self):
        self.model_path = config['model_paths']['label_detector']
        self.confidence = config['confidence']
        self.model = None
        self.set_up_model()

    def set_up_model(self):
        """
        Load the YOLOv8 model for label detection.
        """
        try:
            self.model = YOLO(self.model_path, self.confidence)
        except Exception as e:
            raise Exception('Error loading model. Please check the model path.')

    def detect_bboxes(self, image_path, save_dir):
        """
        Detect bounding boxes around labels in an image and save the cropped labels.

        Args:
            image_path (str): Path to the input image.
            save_dir (str): Directory to save the cropped label images.

        Returns:
            list: Paths of the cropped label images.
            int: Number of detected labels.
        """
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size
        results = self.model(source=image_path, show=False, conf=self.confidence, save=False)

        cropped_image_paths = []
        amount_of_labels = []

        try:
            for r in results:
                boxes = r.boxes.cpu().numpy()
                for i, xyxy in enumerate(boxes.xyxy):
                    crop_box = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                    label_crop = original_image.crop(crop_box)
                    amount_of_labels.append(label_crop)


                    white_background = Image.new("RGB", (original_width, original_height), "White")
                    white_background.paste(label_crop, (int(xyxy[0]), int(xyxy[1])))

                    # Save each cropped image
                    original_image_name = os.path.basename(image_path)
                    file_name, file_ext = os.path.splitext(original_image_name)
                    cropped_image_path = os.path.join(save_dir, f"{file_name}_crop_{i}{file_ext}")
                    white_background.save(cropped_image_path)
                    cropped_image_paths.append(cropped_image_path)

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")

        return cropped_image_paths, len(amount_of_labels)

# OCR Component
class OCRComponent:
   """
   Class for performing Optical Character Recognition (OCR) on images.
   """

   def __init__(self):
       self.transcription_model_weight_path = config['model_paths']['ocr_model']

   def detect_text(self, image_path):
       """
       Detect and return the text present in an image using the OCR pipeline.

       Args:
           image_path (str): Path to the input image.

       Returns:
           str: Detected text from the image.
       """
       return pipe.htrocr_usage(pipe,
                                image_path,
                                transcription_model_weight_path=self.transcription_model_weight_path,
                                save_images=False,
                                out_type='txt')

   def clean_text(self, image_path, output_path="out/None_result.txt"):
       """
       Clean the detected text by removing line numbers and other artifacts.

       Args:
           image_path (str): Path to the input image.
           output_path (str, optional): Path to save the cleaned text. Defaults to "out/None_result.txt".

       Returns:
           str: Cleaned text from the image.
       """
       self.detect_text(image_path)
       clean_texts = []
       with open(output_path, 'r') as file:
           for line in file.readlines():
               regex = re.compile(r"line_\d+\.jpg\t")
               line = regex.sub("", line)
               clean_texts.append(line.strip())
       return ' '.join(clean_texts)

# NER Component
class NERComponent:
   """
   Class for Named Entity Recognition (NER) using the spaCy library.
   """

   def __init__(self):
       self.ner_model_path = config['model_paths']['ner_model']
       self.nlp = spacy.load(self.ner_model_path)

   def detect_entities(self, text):
       """
       Detect named entities in the given text.

       Args:
           text (str): Input text.

       Returns:
           list: List of tuples containing the entity text and its label.
       """
       doc = self.nlp(text)
       entities = [(ent.text, ent.label_) for ent in doc.ents]
       return entities

   def displacy_entities(self, text):
       """
       Display the named entities in the given text using the displaCy visualizer.

       Args:
           text (str): Input text.
       """
       doc = self.nlp(text)
       displacy.serve(doc, style="ent", port=config['displacy_port'])

   def ner_of_txt(self, txt_path):
       """
       Perform Named Entity Recognition on the text in a file.

       Args:
           txt_path (str): Path to the text file.

       Returns:
           list: List of tuples containing the entity text and its label.
       """
       with open(txt_path, 'r') as file:
           text = file.read()
           return self.detect_entities(text)

# Database Component
class Database:
    """
    Class for managing a database of extracted information.
    """

    def __init__(self):
        self.columns = ["Image", "Header", "Plant", "Person", "Date", "Location", "Latitude", "Longitude"]
        self.df = pd.DataFrame(columns=self.columns)

    def add_row(self, image, header, plant, person, date, loc, lat, lon):
        """
        Add a new row to the database.

        Args:
            image (str): Path to the image.
            header (str): Header text.
            plant (str): Plant name.
            person (str): Person name.
            date (str): Date.
            loc (str): Location.
            lat (str): Latitude.
            lon (str): Longitude.
        """
        new_row = pd.DataFrame({
            "Image": [image],
            "Header": [header],
            "Plant": [plant],
            "Person": [person],
            "Date": [date],
            "Location": [loc],
            "Latitude": [lat],
            "Longitude": [lon]
        }, columns=self.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def save(self, save_dir, format='csv'):
        """
        Save the database to a file.
        Args:
            save_dir (str): Directory to save the output file.
            format (str, optional): Output file format ('csv' or 'xlsx'). Defaults to 'csv'.
        """
        if format == 'csv':
            self.df.to_csv(os.path.join(save_dir, "output.csv"), index=False)

    def _print_df(self):
        """
        Print the database content.
        """
        print(self.df)

# Helper Functions
def ocr_label(ocr_component, cropped_image_path):
   """
   Perform OCR on a cropped label image and return the text.

   Args:
       ocr_component (OCRComponent): Instance of the OCRComponent class.
       cropped_image_path (str): Path to the cropped label image.

   Returns:
       str: Detected text from the label image.
   """
   return ocr_component.clean_text(cropped_image_path)

def process_image(image_path, label_detector, ocr_component):
    """
    Process an image by detecting labels, cropping them, and performing OCR on the cropped labels.

    Args:
        image_path (str): Path to the input image.
        label_detector (LabelDetector): Instance of the LabelDetector class.
        ocr_component (OCRComponent): Instance of the OCRComponent class.
        temp_dir (str): Path to the temporary directory for storing cropped images.

    Returns:
        list: List of detected text from the cropped labels.
    """
    cropped_image_paths, amount_of_labels = label_detector.detect_bboxes(image_path, TEMP_DIR)
    texts = []

    if amount_of_labels >= 5:
        amount_of_labels = math.ceil(amount_of_labels / 2)

    # Limit the number of processes to the number of available CPU cores
    max_workers = min(amount_of_labels, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(ocr_label, ocr_component, path): path for path in cropped_image_paths}

        for future in future_to_image:
            try:
                text = future.result()
                texts.append(text)
            except Exception as e:
                logging.error(f"Error processing image {future_to_image[future]}: {e}")

    return texts



# Main Pipeline
class Pipeline:
    """
    Main pipeline class for processing images and extracting information.
    """
    def __init__(self):
        self.label_detector = LabelDetector()
        self.ocr_component = OCRComponent()
        self.ner_component = NERComponent()

        self.db = Database()
        
    def update_database(self, select_random_image=False, download_amount=1, image_path=None):
        """
        Update the database with information extracted from images.

        Args:
            select_random_image (bool, optional): Flag to select random images for processing. Defaults to False.
            download_amount (int, optional): Number of random images to download. Defaults to 1.
            image_path (str, optional): Path to the input image. Defaults to None.
        """
        save_dir = config['output_directory']
        os.makedirs(save_dir, exist_ok=True)

        if select_random_image:
            image_downloader = ImageDownloader("IMAGES")
            image_downloader.download_random_images(download_amount)
            image_paths = [os.path.join("IMAGES", img) for img in os.listdir("IMAGES")]
        else:
            image_paths = [image_path]
        

        
        for img_path in image_paths:
            texts = process_image(img_path, self.label_detector, self.ocr_component)
            
            for text in texts:
                header = ""
                plant = ""
                person = ""
                date = ""
                loc = ""
                lat = ""
                lon = ""
                entities = self.ner_component.detect_entities(text)
                for entity, label in entities:
                    if label == "HEADER":
                        header = entity
                    elif label == "PLANT":
                        plant = entity
                    elif label == "PERSON":
                        person = entity
                    elif label == "DATE":
                        date = entity
                    elif label == "LOC":
                        loc = entity
                    elif label == "LAT":
                        lat = entity
                    elif label == "LON":
                        lon = entity
                self.db.add_row(os.path.basename(img_path), header, plant, person, date, loc, lat, lon)
        self.db.save(save_dir, format='csv')

    def main(self, image_path=None):
        """
        Main function to run the pipeline.

        Args:
            image_path (str, optional): Path to the input image. If not provided, a random image will be downloaded.
        """
        if image_path:
            self.update_database(select_random_image=False, image_path=image_path)
        else:
            self.update_database(select_random_image=True, download_amount=1)

    

    def cleanup(self):
        """
        Clean up temporary files and directories.
        """
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                os.remove(os.path.join(TEMP_DIR, file))
            os.rmdir(TEMP_DIR)

        if os.path.exists("out"):
            for file in os.listdir("out"):
                os.remove(os.path.join("out", file))
            os.rmdir("out")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image or download a random image.')
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.main(image_path=args.image_path)
    pipeline.cleanup()