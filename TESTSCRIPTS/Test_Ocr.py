import random
import easyocr
import re
import matplotlib.pyplot as plt
import numpy as np
from Levenshtein import ratio  # Import the ratio function to calculate similarity
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
from PIL import Image, ImageDraw
import OCR_Module as ocr_module
import os
from PIL import Image
import easyocr
import time
from ultralytics import YOLO
from thefuzz import fuzz
# This module contains tests for Linas Ocr and easyocr and compares the results
# from the two different OCR models. The tests are done on the same data set, we 
# test on Linas handread dataset - Linas100
# model = easyocr.Reader(['en','da',"lt"])

model = YOLO("../Weights/label_detector.pt")

def crop_image(image_path, cropped_photos_dir):

    while True:
        # This method now assumes image_path is always a path to an image file, not a directory
        results = model(source=image_path, show=True, conf=0.5, save=True)
        original_image = Image.open(image_path)
        counter = 0
        cropped_image_path = None  # Initialize cropped_image_path with a default value

        for r in results:
            boxes = r.boxes.cpu().numpy()
            xyxys = boxes.xyxy

            for i, xyxy in enumerate(xyxys):
                label_name = model.names[int(boxes.cls[i])]
                original_image_name = os.path.basename(image_path)  # Get the filename from the image path
                original_image_name_without_ext = os.path.splitext(original_image_name)[0]  # Remove the extensio
                cropped_image_path = os.path.join(cropped_photos_dir, original_image_name_without_ext + f".att.jpg")
                cropped_image = original_image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                cropped_image.save(cropped_image_path)

    return cropped_image_path
 
    
    
def crop_image1(image_path, save_dir):
    """
    This function renders the background of the image white, such that we can
    use the OCR model to detect text on the true image and not the cropped image
    """
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    eggshell_color = (240, 234, 214)  # RGB representation of eggshell color
    white_background = Image.new("RGB", (original_width, original_height), eggshell_color)
    # Create a white image of the same size as the original image
    
    
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

def crop_image2(image_path, save_dir):
    """
    This function renders the background of the image white, such that we can
    use the OCR model to detect text on the true image and not the cropped image
    """
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size
    eggshell_color = (255, 255, 255)  # RGB representation of eggshell color
    white_background = Image.new("RGB", (original_width, original_height), eggshell_color)
    # Create a white image of the same size as the original image
    
    
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


def group_texts_by_y_coordinate(results, y_tolerance=15):
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
    This functiin reads the results from the Linas OCR model and returns the detected text
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
    
def find_true_values(image_path, data_dir = "100.xlsx"):
    """
    Fetches the true values from the dataset for the given image
    """
    df = pd.read_excel(data_dir).dropna()
    true_values = []
    for index, row in df.iterrows():
        if row["image"] == os.path.basename(image_path):
            true_values.append(row["text"])
    return true_values
    # return true_values

def normalize_text(text):
    """
    This function normalizes the text by removing special characters and whitespaces
    """
    
    return re.sub(r'\W+', ' ', text).strip(".")

def test_accuracy_easy(image_path):
    """
    This function tests the accuracy of the Easy OCR model on the given image 
    and returns the accuracy in percentage. It matches each OCR word to the closest true word without repetition.
    """
    easy_ocr_results = [normalize_text(result) for result in test_easy_ocr(f"{image_path}")]
    ocr_words = [word for result in easy_ocr_results for word in result.split()]
    true_values = [normalize_text(value) for value in find_true_values(f"{image_path}")]
    true_words = [word for value in true_values for word in value.split()]

    print("OCR Results:", easy_ocr_results)
    print("OCR Words:", ocr_words)

    if not true_words:
        print(f"No true words found for {os.path.basename(image_path)}. Cannot compute accuracy.")
        return [0]  # Return a list with 0 to maintain the structure used in your plotting function

    matched_words = set()  # To keep track of which OCR words have been matched
    word_count = 0
    list_of_accs = []

    for true_word in true_words:
        best_match = None
        highest_score = 0
        for ocr_word in ocr_words:
            if ocr_word not in matched_words:  # Only consider unmatched OCR words
                score = fuzz.ratio(true_word, ocr_word)
                if score > highest_score:
                    highest_score = score
                    best_match = ocr_word
        if highest_score > 70:  # Using a similarity threshold of 70%
            word_count += 1
            matched_words.add(best_match)  # Mark this OCR word as matched

    acc = (word_count / len(true_words)) * 100
    list_of_accs.append(acc)
    print("Matched OCR Words:", matched_words)
    return list_of_accs

def test_accuracy_linas(image_path, locr_results, levenshtein_threshold=0.5):
    """
    This function tests the accuracy of the Linas OCR model on the given image
    and returns the accuracy in percentage. It matches each OCR word to the closest true word without repetition.
    """
    print("----------------------------------------------------------------")
    print("Words found by the algorithm", locr_results)
    locr_words = [word for result in locr_results for word in result.split()]
    true_values = [normalize_text(value) for value in find_true_values(f"{image_path}")]
    true_words = [word for value in true_values for word in value.split()]
    print("The True words are", true_words)  

    if len(true_words) == 0:
        print(f"No true words found for {os.path.basename(image_path)}. Cannot compute accuracy.")
        return [0]  # Return a list with 0 to maintain the structure used in your plotting function

    print(f"compute {os.path.basename(image_path)}")
    matched_words = set()  # To keep track of which OCR words have been matched
    word_count = 0
    list_of_accs = []

    for true_word in true_words:
        best_match = None
        highest_score = 0
        for ocr_word in locr_words:
            if ocr_word not in matched_words:
                score = fuzz.ratio(true_word, ocr_word)
                if score > highest_score:
                    highest_score = score
                    best_match = ocr_word
        if highest_score >= levenshtein_threshold * 100:  # Adjusting the threshold for fuzz.ratio
            word_count += 1
            matched_words.add(best_match)  # Mark this OCR word as matched

    acc = (word_count / len(true_words)) * 100 if len(true_words) > 0 else 0
    list_of_accs.append(acc)
    print("Matched OCR words:", matched_words)
    print("----------------------------------------------------------------") 
    return list_of_accs

# Assume necessary OCR and image cropping functions are defined elsewhere

def fetch_random_image(path):
    
    random_image = random.choice(os.listdir(path))
    return random_image


    
def plot_accuracy_comparison(easy_ocr_accuracies, linas_ocr_accuracies, filenames, average_easy, average_linas):
    # Flatten the lists of accuracies
    easy_ocr_accuracies = [acc for sublist in easy_ocr_accuracies for acc in sublist]
    linas_ocr_accuracies = [acc for sublist in linas_ocr_accuracies for acc in sublist]
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Number of pairs of bars
    num_pairs = len(filenames)
    index = np.arange(num_pairs)  # This will be the positions of the tick labels (center between bars)
    bar_width = 0.35
    
    # Plot the bars
    ax.bar(index - bar_width/2, easy_ocr_accuracies, bar_width, label='EasyOCR', color='skyblue')
    ax.bar(index + bar_width/2, linas_ocr_accuracies, bar_width, label='LinasOCR', color='lightgreen')

    # Plot average accuracy lines
    ax.axhline(y=average_easy, color='blue', linestyle='--', linewidth=2, label='EasyOCR Accuracy')
    ax.axhline(y=average_linas, color='green', linestyle='--', linewidth=2, label='LinasOCR Accuracy')

    # Set labels for axes and title
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy: Linas OCR vs Easy OCR')

    # Assign x-tick labels
    ax.set_xticks(index)  # Position labels between the bars
    accuracy_labels = [f'{easy_acc:.2f}% | {linas_acc:.2f}%' for easy_acc, linas_acc in zip(easy_ocr_accuracies, linas_ocr_accuracies)]
    ax.set_xticklabels(accuracy_labels, rotation=45, ha="right")

    ax.legend()
    plt.tight_layout()
    plt.savefig("LinasVSEasy.png")
    plt.show()


if __name__ == "__main__":
    cropped_image = crop_image("Linas100/sp6592993101004696521.jpg", "SchnellesCroppen")
    
    # cropped_images_dir = "Linas100_cropped"
    # white_cropped_images_dir = "white_cropped"
    # white_cropped = []
    # normal_cropped = []
    # filenames_list = []
    # iteration_count = 0
    # max_iterations = 15
    
    
    # for filename in os.listdir("Linas100"):
    #     if iteration_count >= max_iterations:
    #         break
    #     if filename == ".DS_Store" or not filename.endswith('.jpg'):
    #         continue
        
    #     base_image_path = filename.rstrip('.jpg') + '.att.jpg'
    #     image_path = os.path.join("Linas100", filename)
    #     cropped_image_path = crop_image(image_path, cropped_images_dir)

    #     ## This code takes the original image as input
    #     # locr_cropped = ocr_module.LinasOCR(image_path)
    #     # locr_cropped.read_text()
    #     # locr_results = test_linas_ocr()
    #     easy_ocr_accuracies = test_accuracy_easy(cropped_image_path)
    #     # linas_ocr_accuracies = test_accuracy_linas(base_image_path, locr_results)
        
        
    #     ## Takes the cropped image as input
    #     # locr_cropped = ocr_module.LinasOCR(cropped_image_path)
    #     # locr_cropped.read_text()
    #     # locr_results = test_linas_ocr()
    #     # # easy_ocr_accuracies = test_accuracy_easy(cropped_image_path)
    #     # linas_ocr_accuracies = test_accuracy_linas(base_image_path, locr_results)
    #     normal_cropped.append(easy_ocr_accuracies)
    #     time.sleep(1)
        
        
    #     ## This code executes on a white cropped image
    #     cropped_image_path2 = crop_image1(image_path, white_cropped_images_dir)
 
    #     locr_not_cropped = ocr_module.LinasOCR(cropped_image_path2)
    #     locr_not_cropped.read_text()
    #     locr_not_cropped = test_linas_ocr()
    #     ## Base image path is the one in the Dataset Linas100   
    #     not_cropped_accuracy = test_accuracy_linas(base_image_path, locr_not_cropped)
    #     white_cropped.append(not_cropped_accuracy)
        
    
    #     filenames_list.append(filename)
        
    #     iteration_count += 1
    #     print("Iteration :" , iteration_count)
        
    # average_linas = np.mean(normal_cropped)
    # average_easy = np.mean(white_cropped)
    
    # print("Average Linas Accuracy", average_linas)
    # print("Average White Cropped OCR Accuracy", average_easy)
    
    # plot_accuracy_comparison(white_cropped, normal_cropped, filenames_list, average_linas, average_easy)
