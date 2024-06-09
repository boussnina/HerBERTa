import os
import requests
import pandas as pd
import cv2
import logging
from collections import Counter
from ultralytics import YOLO

# Constants
WEIGHTS1 = "WEIGHTS/label_detector.pt"
WEIGHTS2 = "WEIGHTS/sheet-component-medium.pt"
IMAGE_DIR = "YOLO_TEST_IMAGES"
SAVE_DIR1 = "Prediction_label_detector"
SAVE_DIR2 = "Prediction_sheet-component-medium"

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR1, exist_ok=True)
os.makedirs(SAVE_DIR2, exist_ok=True)

class TestYOLOv8Models:
    def __init__(self, model):
        self.confidence = 0.47
        if model == 1:
            self.yolo = YOLO(WEIGHTS1)
        elif model == 2:
            self.yolo = YOLO(WEIGHTS2)
        else:
            raise ValueError("Invalid model number")

    def detect_bboxes(self, image_path, save_dir):
        results = self.yolo(source=image_path, show=False, conf=self.confidence, save=True)
        image_name = os.path.basename(image_path)
        os.makedirs(save_dir, exist_ok=True)
        for result in results:
            if not hasattr(result, 'boxes'):
                continue
            annotated_img = result.plot()
            output_path = os.path.join(save_dir, image_name)
            cv2.imwrite(output_path, annotated_img)

    def count_from_model(self, image_path):
        results = self.yolo(source=image_path, show=False, conf=self.confidence, save=True)
        labels = []
        for result in results:
            for box in result.boxes:
                class_id = box.cls.item()
                class_name = self.yolo.names[int(class_id)]
                labels.append(class_name)
        return len(labels)

    def count_institutional(self, image_path):
        results = self.yolo(source=image_path, show=False, conf=self.confidence, save=False)
        count_institutional = 0
        for result in results:
            for box in result.boxes:
                class_id = box.cls.item()
                class_name = self.yolo.names[int(class_id)]
                if class_name.startswith("institutional"):
                    count_institutional += 1
        return count_institutional

def create_results_dataframe(model1, model2, image_dir):
    results = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        labels_model1 = model1.count_from_model(image_path)
        institutional_count_model2 = model2.count_institutional(image_path)
        results.append({
            'image_name': image_name,
            'labels_model1': labels_model1,
            'institutional_count_model2': institutional_count_model2
            
        })
    return pd.DataFrame(results)


def calculate_detection_accuracy(detected_csv, true_csv):
    detected_df = pd.read_csv(detected_csv)
    true_df = pd.read_csv(true_csv)
    true_dict = true_df.set_index('image_name')['true_count'].to_dict()

    total_images = 0
    correctly_detected_labels_model1 = 0
    correctly_detected_institutional_count_model2 = 0
    false_positives_model1 = 0
    false_positives_model2 = 0

    for _, row in detected_df.iterrows():
        image_name = row['image_name']
        labels_model1 = row['labels_model1']
        institutional_count_model2 = row['institutional_count_model2']

        if image_name not in true_dict:
            print(f"No true count data found for {image_name}. Skipping...")
            continue

        true_count = true_dict[image_name]
        total_images += 1

        if labels_model1 == true_count:
            correctly_detected_labels_model1 += 1
        else:
            false_positives_model1 += abs(labels_model1 - true_count)

        if institutional_count_model2 == true_count:
            correctly_detected_institutional_count_model2 += 1
        else:
            false_positives_model2 += abs(institutional_count_model2 - true_count)

    accuracy_labels_model1 = (correctly_detected_labels_model1 / total_images) * 100 if total_images > 0 else 0
    accuracy_institutional_count_model2 = (correctly_detected_institutional_count_model2 / total_images) * 100 if total_images > 0 else 0

    return {
        "total_images": total_images,
        "accuracy_labels_model1": accuracy_labels_model1,
        "accuracy_institutional_count_model2": accuracy_institutional_count_model2,
        "false_positive_rate_labels_model1": (false_positives_model1 / total_images) * 100 if total_images > 0 else 0,
        "false_positive_rate_institutional_count_model2": (false_positives_model2 / total_images) * 100 if total_images > 0 else 0
    }

if __name__ == "__main__":
    model1 = TestYOLOv8Models(1)
    model2 = TestYOLOv8Models(2)

    for image_name in os.listdir(IMAGE_DIR):
        if image_name.startswith('DS_Store'):
            continue

        image_path = os.path.join(IMAGE_DIR, image_name)
        model1.detect_bboxes(image_path, save_dir=SAVE_DIR1)
        model2.detect_bboxes(image_path, save_dir=SAVE_DIR2)

    results_df = create_results_dataframe(model1, model2, IMAGE_DIR)
    results_df.to_csv('results.csv', index=False)

    hand_counted_csv = 'real_count.csv'
    accuracy_results = calculate_detection_accuracy('results.csv', hand_counted_csv)

    
    
   
