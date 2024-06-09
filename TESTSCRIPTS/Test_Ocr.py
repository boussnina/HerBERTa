import random
import easyocr
import re
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image, ImageDraw
import os
from PIL import Image
import easyocr
import time
from ultralytics import YOLO
from thefuzz import fuzz
from htrocr.run import NHMDPipeline as pipe
import logging

class LinasOCR:
    def __init__(self, image_path):
        self.image_path = image_path

    def read_text(self):
        # Use PyTesseract to perform OCR on the image
        return pipe.htrocr_usage(pipe, self.image_path, transcription_model_weight_path="/content/drive/MyDrive/TrainSpacy/OCR_testing/nhmd_base_full_final")

# This module contains tests for Linas Ocr and easyocr and compares the results
# from the two different OCR models. The tests are done on the same data set, we
# test on Linas handread dataset - Linas100
# model = easyocr.Reader(['en','da',"lt"])

model = YOLO("../WEIGHTS/label_detector.pt")

def crop_image(image_path, cropped_photos_dir):
    results = model(source=image_path, show=False, conf=0.5, save=False)
    original_image = Image.open(image_path)
    cropped_image_path = None
    if len(results) == 0:
        print(f"No labels detected in {image_path}. Skipping cropping.")
        return None

    for r in results:
        boxes = r.boxes.cpu().numpy()
        xyxys = boxes.xyxy

        for i, xyxy in enumerate(xyxys):
            label_name = model.names[int(boxes.cls[i])]
            original_image_name = os.path.basename(image_path)
            original_image_name_without_ext = os.path.splitext(original_image_name)[0]
            cropped_image_path = os.path.join(cropped_photos_dir, original_image_name_without_ext + f".att.jpg")
            cropped_image = original_image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
            cropped_image.save(cropped_image_path)

        if cropped_image_path == None:
          print(image_path)
    return cropped_image_path

def crop_and_adjust_background(image_path, save_dir):
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    eggshell_color = (255, 255, 255)  # RGB representation of eggshell color
    white_background = Image.new("RGB", (original_width, original_height), eggshell_color)

    # Detect labels
    results = model(source=image_path, show=False, conf=0.4, save=False)
    draw = ImageDraw.Draw(white_background)

    for r in results:
        boxes = r.boxes.cpu().numpy()
        for xyxy in boxes.xyxy:
            # Draw the detected label from the original image onto the white background
            crop_box = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            label_crop = original_image.crop(crop_box)
            white_background.paste(label_crop, (int(xyxy[0]), int(xyxy[1])))

    # Save the modified image
    original_image_name = os.path.basename(image_path)
    modified_image_path = os.path.join(save_dir, f"white_bg_{original_image_name}")
    white_background.save(modified_image_path)
    return modified_image_path

def group_texts_by_y_coordinate(results, y_tolerance=70):
    lines = []
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        avg_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
        added_to_line = False
        for line in lines:
            if abs(line['avg_y'] - avg_y) <= y_tolerance:
                line['texts'].append(text)
                line['avg_y'] = (line['avg_y'] * len(line['texts']) + avg_y) / (len(line['texts']) + 1)  # Recalculate average y
                added_to_line = True
                break
        if not added_to_line:
            lines.append({'avg_y': avg_y, 'texts': [text]})
    concatenated_lines = [' '.join(line['texts']) for line in lines]
    return concatenated_lines

def test_easy_ocr(image_path):
    """
    This function tests the Easy OCT model on the cropped image and
    returns the detected text

    It does not provide good results when run a the whole image
    """
    reader = easyocr.Reader(['en','da',"lt"])
    result = reader.readtext(image_path)
    detected_text = group_texts_by_y_coordinate(result)
    return detected_text

def test_linas_ocr():
    """
    This function reads the results from the Linas OCR model and returns the detected text
    Linas OCR returns a txt file with the detected text in the format:
    line_1.jpg    text1
    line_2.jpg    text2
    This Function reads the text and returns a list of the detected text
    """
    with open("out/None_result.txt", "r") as file:
        data = file.read().replace('\n', ' ')
        clean_text = []
        for line in data.split(" "):
            regex = re.compile(r"line_\d+.jpg\t")
            line = regex.sub("", line)
            clean_text.append(line)
        return clean_text

def find_true_values(image_path, data_dir = "../CSV_FILES"):
    """
    Fetches the true values from the dataset for the given image
    """
    print("this is the true image path :", image_path)
    df = pd.read_excel(data_dir).dropna()
    true_values = []

    for index, row in df.iterrows():

        if row["image"] == os.path.basename(image_path):
            true_values.append(row["text"])
    return true_values

def normalize_text(text):
    """
    This function normalizes the text by removing special characters and whitespaces
    """
    return re.sub(r'\W+', ' ', text).strip(".")

def test_accuracy_easy(image_path, threshold=70):
    easy_ocr_results = [normalize_text(result) for result in test_easy_ocr(f"{image_path}")]
    ocr_words = [word for result in easy_ocr_results for word in result.split()]
    true_values = [normalize_text(value) for value in find_true_values(f"{image_path}")]
    true_words = [word for value in true_values for word in value.split()]

    print(ocr_words)
    print(true_words)
    if not true_values:
        return 0, 0, 0

    tp = 0
    fp = 0
    matched_words = set()

    for ocr_word in ocr_words:
        best_match = None
        highest_score = 0
        for true_word in true_words:
            score = fuzz.ratio(true_word, ocr_word)
            if score > highest_score:
                highest_score = score
                best_match = true_word
        if highest_score > threshold:
            if best_match not in matched_words:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1

    fn = len(true_words) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    substitutions = len(matched_words)
    deletions = len(true_words) - substitutions
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


    return precision, recall, f1

def test_accuracy_linas(image_path, locr_results, threshold=70):
    print("enter")
    locr_words = [word for result in locr_results for word in result.split()]
    true_values = [normalize_text(value) for value in find_true_values(f"{image_path}")]
    true_words = [word for value in true_values for word in value.split()]
    print(locr_words)
    print(true_words)

    if len(true_words) == 0:
        return 0, 0, 0, 0

    tp = 0
    fp = 0
    matched_words = set()

    for ocr_word in locr_words:
        best_match = None
        highest_score = 0
        for true_word in true_words:
            score = fuzz.ratio(true_word, ocr_word)
            if score > highest_score:
                highest_score = score
                best_match = true_word
        if highest_score > threshold:
            if best_match not in matched_words:
                matched_words.add(best_match)
                tp +=1
            else:
                fp +=1
        else:
            fp +=1

    fn = len(true_words) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    substitutions = len(matched_words)
    deletions = len(true_words) - substitutions
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":


    white_cropped_images_dir = "White_cropped"
    if not os.path.exists(white_cropped_images_dir):
        print("There is no Directory")
    white_cropped_precision_linas = []
    white_cropped_recall_linas = []
    white_cropped_f1_linas = []
    white_cropped_precision_easy = []
    white_cropped_recall_easy = []
    white_cropped_f1_easy = []
    filenames_list = []
    iteration_count = 0

    try:
        for filename in os.listdir("15CompLables"):
            print(filename)


            if filename == ".DS_Store" or not filename.endswith('.jpg'):
                continue

            base_image_path = filename.rstrip('.jpg') + '.att.jpg'
            image_path = os.path.join("15CompLables", filename)
            print(image_path)
            cropped_image_path = crop_and_adjust_background(image_path, white_cropped_images_dir)
            print(cropped_image_path)
            normal_crop_image = crop_image(image_path, "Linas100_cropped")
            if cropped_image_path is None:
                continue  # Skip further processing if no labels were detected

            print("LINAS OCR START")
            locr_white_cropped = LinasOCR(cropped_image_path)
            locr_white_cropped.read_text()
            locr_white_cropped_results = test_linas_ocr()
            print(locr_white_cropped_results)
            white_cropped_precision_linas_val, white_cropped_recall_linas_val, white_cropped_f1_linas_val = test_accuracy_linas(base_image_path, locr_white_cropped_results)
            white_cropped_precision_linas.append(white_cropped_precision_linas_val)
            white_cropped_recall_linas.append(white_cropped_recall_linas_val)
            white_cropped_f1_linas.append(white_cropped_f1_linas_val)


            print("EASY OCR START")
            easy_ocr_precision, easy_ocr_recall, easy_ocr_f1 = test_accuracy_easy(normal_crop_image)
            white_cropped_precision_easy.append(easy_ocr_precision)
            white_cropped_recall_easy.append(easy_ocr_recall)
            white_cropped_f1_easy.append(easy_ocr_f1)
            filenames_list.append(filename)
            iteration_count += 1

            print("Iteration:", iteration_count)
    except Exception as e:
        print(e)
        print("Error in processing image")
        raise Exception("Error in processing image")
    
    average_linas_precision = np.mean(white_cropped_precision_linas)
    average_linas_recall = np.mean(white_cropped_recall_linas)
    average_linas_f1 = np.mean(white_cropped_f1_linas)
    average_easy_precision = np.mean(white_cropped_precision_easy)
    average_easy_recall = np.mean(white_cropped_recall_easy)
    average_easy_f1 = np.mean(white_cropped_f1_easy)



    print("Average Linas Precision (White Cropped):", average_linas_precision)
    print("Average Linas Recall (White Cropped):", average_linas_recall)
    print("Average Linas F1 Score (White Cropped):", average_linas_f1)

    print("Average Easy OCR Precision (White Cropped):", average_easy_precision)
    print("Average Easy OCR Recall (White Cropped):", average_easy_recall)
    print("Average Easy OCR F1 Score (White Cropped):", average_easy_f1)