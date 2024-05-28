import pandas as pd
import json
import re
import os
from PIL import Image, ImageDraw
import spacy
from spacy import displacy
from ultralytics import YOLO
from htrocr.run import NHMDPipeline as pipe

TEMP_DIR = "tempory_files"#
IMAGE_DIR = "IMAGES"


if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

class LabelDetector:
    def __init__(self):
        self.model_path = config['model_paths']['label_detector']
        self.confidence = config['confidence']
        self.model = None
        self.set_up_model()
    
    def set_up_model(self):
        try:
            self.model = YOLO(self.model_path, self.confidence)
        except Exception as e:
           
            raise Exception('Error loading model. Please check the model path.')
    
    def detect_bboxes(self, image_path, save_dir):
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size
        white_background = Image.new("RGB", (original_width, original_height), "White")
        results = self.model(source=image_path, show=False, conf=self.confidence, save=False)
        draw = ImageDraw.Draw(white_background)
        if len(results) > 0:
                
            for r in results:
            
                boxes = r.boxes.cpu().numpy()
                for xyxy in boxes.xyxy:
                    crop_box = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                    label_crop = original_image.crop(crop_box)
                    white_background.paste(label_crop, (int(xyxy[0]), int(xyxy[1])))

            original_image_name = os.path.basename(image_path)
            modified_image_path = os.path.join(save_dir, f"cropped_{original_image_name}")
            white_background.save(modified_image_path)
            return modified_image_path
        else:
            return None
            

class OCRComponent:
    def __init__(self):
        self.transcription_model_weight_path = config['model_paths']['ocr_model']
        
    def detect_text(self, image_path):
        return pipe.htrocr_usage(pipe, 
                                 image_path, 
                                 transcription_model_weight_path=self.transcription_model_weight_path, 
                                 save_images=False, 
                                 out_type='txt')

    def clean_text(self, image_path, output_path="out/None_result.txt"):
        self.detect_text(image_path)
        clean_texts = []
        with open(output_path, 'r') as file:
            for line in file.readlines():
                regex = re.compile(r"line_\d+\.jpg\t")
                line = regex.sub("", line)
                clean_texts.append(line.strip())  
        return ' '.join(clean_texts)

class NERComponent: 
    def __init__(self):
        self.ner_model_path = config['model_paths']['ner_model']
        self.nlp = spacy.load(self.ner_model_path)

    def detect_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def displacy_entities(self, text):
        doc = self.nlp(text)
        displacy.serve(doc, style="ent", port=config['displacy_port'])
        
    def ner_of_txt(self, txt_path):
        with open(txt_path, 'r') as file:
            text = file.read()
            return self.detect_entities(text)
        
        
class Database:
    def __init__(self):
        self.columns = ["Image", "Header", "Plant", "Person", "Date", "Location", "Latitude", "Longitude"]
        self.df = pd.DataFrame(columns=self.columns)
        
    def add_row(self, image, header, plant, person, date, loc, lat, lon):
        # Create a new DataFrame for the row to be added
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
        if format == 'csv':
            self.df.to_csv(os.path.join(save_dir, "output.csv"), index=False)
    
    def _print_df(self):
        print(self.df)
        
        
def process_image(image_path, label_detector, ocr_component):
    try:
        modified_image_path = label_detector.detect_bboxes(image_path, TEMP_DIR)
        if modified_image_path is None:
            return []
        texts = ocr_component.clean_text(modified_image_path)
        return texts
    finally:
        if os.path.exists(modified_image_path):
            os.remove(modified_image_path)
            
def main():
    save_dir = config['output_directory']
    os.makedirs(save_dir, exist_ok=True)
    
    label_detector = LabelDetector()
    ocr_component = OCRComponent()
    ner_component = NERComponent()
    db = Database()

    # Assume multiple images to process
    image_paths = [os.path.join("IMAGES", img) for img in os.listdir("IMAGES")]

    for img_path in image_paths:
        texts = process_image(img_path, label_detector, ocr_component)
        entities = ner_component.detect_entities(texts)
        header = ""
        plant = ""
        person = ""
        date = ""
        loc = ""
        lat = ""
        lon = ""

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
        db.add_row(os.path.basename(img_path), header, plant, person, date, loc, lat, lon)
    db.save(save_dir)
    db._print_df()

def cleanup():
    if os.path.exists(TEMP_DIR) :
        os.rmdir(TEMP_DIR)
    
    if os.path.exists("out"):
        os.remove("out/None_result.txt")
        os.rmdir("out")
        
    
if __name__ ==  "__main__":
    main()
    cleanup()
    

