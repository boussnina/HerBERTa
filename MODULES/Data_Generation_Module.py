from datetime import datetime, timedelta
import random
import pandas as pd
import spacy
import re
import json
import spacy
from spacy.tokens import DocBin
import json

spacy.prefer_gpu()


def number_to_month(number):
    months = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    return months.get(number, "Invalid number")

def int_to_roman(num):
    roman_numerals = {
        1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L',
        90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'
    }
    result = ''
    for value in sorted(roman_numerals.keys(), reverse=True):
        while num >= value:
            result += roman_numerals[value]
            num -= value
    return result

def get_random_date(min_year=1800):
    min_year = datetime(year=min_year, month=1, day=1)
    current_year = datetime.now()

    total_days = (current_year - min_year).days

    random_number_of_days = random.randint(0, total_days)

    random_date = min_year + timedelta(days=random_number_of_days)

    return random_date

def randomize_date_format(random_date):
    probability = random.randint(1,110)
    #94
    if probability > 94:
        formatted_date = str(random_date.day) + "/" + str(random_date.month) + " " + str(random_date.year)
    #80
    elif probability > 80:
        formatted_date = str(random_date.year) + ", " + number_to_month(random_date.month) + " " + str(random_date.day)
    #67
    elif probability > 67: 
        formatted_date = "Date " + str(random_date.year) + ", " + number_to_month(random_date.month) + " " + str(random_date.day)
    #57
    elif probability > 57:
        formatted_date = str(random_date.year)
    # 50
    elif probability > 50:
        formatted_date = "Date: " + str(random_date.day) + "/" + str(random_date.month) + "/" + str(random_date.year)
    # 44
    elif probability > 44:
        formatted_date = str(random_date.day) + ". " + number_to_month(random_date.month) + " " + str(random_date.year)
    #39
    elif probability > 39:
        formatted_date = "Date "  + str(random_date.day) + "-" + int_to_roman(random_date.month) + "-" + str(random_date.year)
    # 35
    elif probability > 35:
        formatted_date = "Date: " + str(random_date.year) + ", " + number_to_month(random_date.month) + " " + str(random_date.day)
    # 31
    elif probability > 31:
        formatted_date = "Date: " + str(random_date.day) + "/" + str(random_date.month) + " " + str(random_date.year)
    # 28
    elif probability > 28:
        formatted_date = str(random_date.day) + "." + str(random_date.month) + "." + str(random_date.year)
    #25
    elif probability > 25:
        formatted_date = str(random_date.day) + "/" + str(random_date.month) + " " + str(random_date.year)[-2:]
    # 22
    elif probability > 22:
        formatted_date = "Date " + number_to_month(random_date.month) + " " + str(random_date.year) + " " + str(random_date.day)
    # 19
    elif probability > 19:
        formatted_date = str(random_date.day) + " " + number_to_month(random_date.month) + " " + str(random_date.year)
    # 17
    elif probability > 17:
        formatted_date = str(random_date.day) + "/" + str(random_date.month)
    # 15
    elif probability > 15:
        formatted_date = str(random_date.day) + "-" + int_to_roman(random_date.month) + "-" + str(random_date.year)
    # 13
    elif probability > 13: 
        formatted_date = "Date " + str(random_date.day) + ". " + str(random_date.month) + ". " + str(random_date.year)
    # 11
    elif probability > 11: 
        formatted_date = "Date "  + str(random_date.day) + "." + str(random_date.month) + "." + str(random_date.year)
    # 10
    elif probability > 10:
        formatted_date = str(random_date.day) + "-"+ str(random_date.month) + "-"+ str(random_date.year)
    # 9
    elif probability > 9:
        formatted_date = str(random_date.day) + " - " + str(random_date.month) + " - " + str(random_date.year)
    # 8
    elif probability > 8: 
        formatted_date = "Date: " + str(random_date.day) + "-" + str(random_date.month) + "-" + str(random_date.year)
    # 7
    elif probability > 7: 
        formatted_date = str(random_date.day) + "." + str(random_date.month) + ". " + str(random_date.year)
    # 6
    elif probability > 6: 
        formatted_date =  "Date " + str(random_date.year) + ", " + number_to_month(random_date.month) + ". " + str(random_date.day)
    # 5
    elif probability > 5:
        formatted_date = number_to_month(random_date.month) + ". " + str(random_date.year) 
    # 4
    elif probability > 4:
        formatted_date = "Date " + str(random_date.year)
    # 3
    elif probability > 3:
        formatted_date = str(random_date.day) + ". " + number_to_month(random_date.month) + ". " + str(random_date.year) + ". "
    # 2
    elif probability > 2:
        formatted_date = "Date " + str(random_date.day) + ". " + number_to_month(random_date.month) + ". " + str(random_date.year) + "."
    # 1
    elif probability > 1: 
        formatted_date = str(random_date.day) +  ". / " + str(random_date.month) + ". " + str(random_date.year)
    else:
        formatted_date = number_to_month(random_date.month) + " " + str(random_date.day) + ", " + str(random_date.year)
    return formatted_date

def get_random_name():
    df = pd.read_csv('../greenland_herb.csv')
    unique_entries = df['1,10,30-collectors.collectingevent.collectors'].unique()
    random_index = random.randint(0, len(unique_entries) - 1)
    random_name = str(unique_entries[random_index])
    # leg., Leg., legit, det., Det., determ.
    title = ["leg.", "leg:", "leg", "Leg.", "Leg:", "Leg", "legit", "det.", 
             "det:", "det", "Det.", "Det:", "Det", "determ."]
    probability = random.randint(0, 100)
    if probability > 25:
        random_idx = random.randint(0, len(title)-1)
        random_name = str(title[random_idx]) + " " + str(random_name)
    elif probability > 24:
        random_index = random.randint(0, len(unique_entries) - 1)
        random_name_2 = unique_entries[random_index]
        random_name = "Leg.: & det.: " + str(random_name) + " & " + str(random_name_2)
    return re.sub(' +', ' ', random_name)

def get_random_location():
    df = pd.read_csv('../greenland_herb.csv')
    unique_entries = df['1,10,2.locality.localityName'].unique()
    random_index = random.randint(0, len(unique_entries) - 1)
    random_location = unique_entries[random_index]
    probability = random.randint(0,91)
    if probability > 47:
        random_location = random_location
    elif probability > 9:
        random_location = f"Loc. {random_location}"
    elif probability > 4:
        random_location = f"Locality {random_location}"
    elif probability > 3:
        random_location = f"Lokalitet {random_location}"
    elif probability > 2:
        random_location = f"Disko: {random_location}"
    elif probability > 1:
        random_location = f"W. {random_location}"
    else:
        random_location = f"Loc.: {random_location}"

    return random_location.strip()

def get_random_plant():
    df = pd.read_csv('../greenland_herb.csv')
    unique_entries = df['1,9-determinations.collectionobject.determinations'].unique()
    random_index = random.randint(0, len(unique_entries) - 1)
    random_plant = unique_entries[random_index]
    if "(current)" in random_plant.lower():
        random_plant = random_plant.replace("(current)", "")
    return random_plant.strip()

def generate_random_coordinates():
    probability = random.randint(0,75)
    lat = random.randint(59, 83)
    lon = random.randint(11, 74) 
    if probability > 74:
        lat, lon = f"Lat. {lat}°{random.randint(0,70)}'", f"Long. {lon}°{random.randint(0,70)}'"
    elif probability > 23:
        lat, lon = f"Lat. {lat}°{random.randint(0,70)}' N.", f"Long. {lon}°{random.randint(0,70)}' W."
    elif probability > 18:
        lat, lon = f"Latitude {lat}°{random.randint(0,70)}' N", f"Longitude {lon}°{random.randint(0,70)}' W"
    elif probability > 14:
        lat, lon = f"lat. {lat}°{random.randint(0,70)}' N", f"long. {lon}°{random.randint(0,70)}' W"
    elif probability > 8:
        lat, lon = f"{lat}°{random.randint(0,70)}' N. lat.", f"{lon}°{random.randint(0,70)}' W. long."
    elif probability > 5:
        lat, lon = f"{lat}°{random.randint(0,70)}' N.", f"{lon}°{random.randint(0,70)}' W."
    elif probability > 3:
        lat, lon = f"{lat}°{random.randint(0,70)}' lat", f"{lon}°{random.randint(0,70)}' long"
    elif probability > 1:
        lat, lon = f"{lat}°{random.randint(0,70)}'", f"{lon}°{random.randint(0,70)}'"
    else:
        lat, lon = f"Lat {lat}°{random.randint(0,70)}'", f"Long {lon}°{random.randint(0,70)}'"
    return lat, lon

def generate_header():
    probability = random.randint(1,157)
    if probability > 134:
        header = "Museum Botanicum Hauniense Grøndlands Botaniske Undersøgelse Plantae Groenlandicae"
    elif probability > 112:
        header = "Museum Botanicum Hauniense Plantae Groenlandicae"
    elif probability > 93:
        header = "Plantae groenlandicae E Museo botanico Hauniensi distributae"
    elif probability > 78:
        header = "DEN DANSKE ARKTISKE STATION DISKO. GRØNLAND"
    elif probability > 68:
        header = "Universitetes botaniske Museum Museum botanicum Hauniense"
    elif probability > 60:
        header = "Dansk geologisk Undersøgelse af Grønland"
    elif probability > 53:
        header = "Botanic Museum of the University, Copenhagen Greenland Plants"
    elif probability > 46:
        header = "Herbarium musei botanici Hauniensis"
    elif probability > 40:
        header = "FLORA GROENLANDICA"
    elif probability > 35:
        header = "Plants from South Greenland (botanical district S)"
    elif probability > 30:
        header = "Universitetes botaniske Museum Museum botanicum Hauniense Plantae Groenlandicae"
    elif probability > 26:
        header = "DEN DANSKE ARKTISKE STATION"
    elif probability > 23:
        header = "Museum Botanicum Hauniense PLANTAE VASCULARES GROENLANDICAE EXSICCATAE"
    elif probability > 20:
        header = "Botanical Museum of the University, Copenhagen THE BOTANICAL EXPEDITION TO WEST GREENLAND"
    elif probability > 17:
        header = "Universitetes botaniske Museum, København E Museo Botanico Hauniense"
    elif probability > 14:
        header = "Museum Botanicum Hauniense"
    elif probability > 12:
        header = "Museum Botanicum Hauniense Plants of West Greenland"
    elif probability > 10:
        header = "Museum Botanicum Hauniense Greenland Botanical Survey"
    elif probability > 8:
        header = "Museum Botanicum Hauniense Universitetets Arktiske Station i Godhavn Plants of West Greenland"
    elif probability > 7:
        header = "Museum Botanicum Hauniense Greenland Botanical Survey Plants of West Greenland"
    elif probability > 6:
        header = "Botanic Museum of the University, Copenhagen Universitetets Arktiske Station i Godhavn Plants of West Greenland"
    elif probability > 5:
        header = "Botanical Museum of the University, Copenhagen Universitetets Arktiske Station i Godhavn Plants of West Greenland"
    elif probability > 4:
        header = "Museum Botanicum Hauniense Kap Farvel Ekspeditionen 1970"
    elif probability > 3:
        header = "Museum Botanicum Hauniense NARSSAQ PROJEKTET Plantae Groenlandicae"
    elif probability > 2:
        header = "FLORA OF GREENLAND JULIANEHAAB DISTRIKT"
    elif probability > 1:
        header = "Museum Botanicum Hauniense Treårsekspeditionen til Christian den X's land"
    else:
        header = "Museum Botanicum Hauniense Plants of Northwest Greenland"
    return header

def generate_catalogue_nr():
    probabilty = random.randint(1,10)
    if probabilty > 9:
        result = "No. " + str(random.randint(0, 99)) + "-" + str(random.randint(0,2000))
    elif probabilty > 8:
        result = "No. " + str(random.randint(0, 99)) + " - " + str(random.randint(0,2000))
    elif probabilty > 7:
        result = "No. " + str(random.randint(0,5000))
    elif probabilty > 6:
        result = "No " + str(random.randint(0, 99)) + "-" + str(random.randint(0,2000))
    elif probabilty > 5:
        result = "No " + str(random.randint(0, 99)) + " - " + str(random.randint(0,2000))
    elif probabilty > 4:
        result = "No " + str(random.randint(0,5000))
    elif probabilty > 3:
        result = str(random.randint(0, 99)) + "-" + str(random.randint(0,2000))
    elif probabilty > 2:
        result = str(random.randint(0, 99)) + " - " + str(random.randint(0,2000))
    else:
        result = str(random.randint(0, 99)) + "-" + str(random.randint(0,2000))
    return result

def generate_area_code():
    probability = random.randint(0,14)
    area_code = ""
    if probability > 13:
        area_code = "S.1"
    elif probability > 12:
        area_code = "N.14"
    elif probability > 6:
        area_code = "W." + str(random.randint(2,7))
    else:
        area_code = "E." + str(random.randint(2,7))
    return area_code
        
def generate_altitude():
    probability = random.randint(1,49)
    prob_0 = random.randint(1,2)
    if probability > 10:
        alt = f"Alt. {random.randint(1,20)}{10^prob_0} m"
    if probability > 1:
        alt = f"Alt. {random.randint(1,20)}{10^prob_0}"
    else:
        alt = f"Altitude {random.randint(1,20)}{10^prob_0}"
    return alt

def generate_spacy_data(label_functions):
    entities_info = [(label, func()) for label, func in label_functions]
    full_text = ""
    entities = []
    start_pos = 0

    for label, value in entities_info:
        if full_text:  
            full_text += " "
            start_pos += 1
        if isinstance(value, float):
            continue
        end_pos = start_pos + len(value)
        entities.append((start_pos, end_pos, label))
        full_text += value
        start_pos = end_pos 

    return (full_text, {"entities": entities})

def generate_label_structure(arg_list):
    lat, lon = generate_random_coordinates()
    label_functions = []
    arg_dict = {1: ("HEADER", generate_header),
                2: ("CATALOGUE_NR", generate_catalogue_nr),
                3: ("AREA_CODE", generate_area_code),
                4: ("PLANT", get_random_plant),
                5: ("LOC", get_random_location),
                6: ("LAT", lambda: lat),
                7: ("LON", lambda: lon),
                8: ("DATE", lambda: randomize_date_format(get_random_date())),
                9:("PERSON", get_random_name),
                10: ("ALT", generate_altitude),
                }
    for argument in arg_list:
        label_functions.append(arg_dict[argument])
    return label_functions

def generate_training_data(x):
    training_data = []
    structures = [
                    [4, 5, 8, 9],
                    [4, 9, 8, 9],
                    [4, 8],
                    [4, 8, 9],
                    [4, 9, 5, 8],
                    [4, 9, 8],
                    [4, 9, 9, 8],
                    [4, 4, 9],
                    [1, 4, 5, 6, 7, 8, 9],
                    [1, 4, 5, 6, 7, 8, 9, 9],
                    [1, 4, 9, 8, 4, 9, 8],
                    [1, 4, 5, 8, 9],
                    [1, 4, 5, 5, 8, 9],
                    [1, 4, 5, 5, 8, 9, 9],
                    [1, 4, 5, 6, 8, 9],
                    [1, 4, 9, 5, 6, 7, 8],
                    [1, 4, 9, 8, 9, 8, 5],
                    [1, 4, 5, 9, 8],
                    [1, 1, 8, 9, 5, 9],
                    [1, 4, 5, 8, 6, 9],
                    # [1, 2, 4, 5, 9, 6, 7, 9, 8, 9],
                    # [1, 2, 4, 5, 6, 7, 10, 8, 9],
                    # [1, 2, 3, 4, 5, 6, 7, 10, 8, 9]
                    ]

    for i in range(x):
        probability = random.randint(0,100)
        if probability > 5:
            select_prob = random.randint(0, len(structures)-1)
            arg_list = structures[select_prob]
        else:
            arg_list = []
            for i in range(len(structures)):
                
                random_seq = random.randint(0, len(structures)-1)
                try:
                    arg_list.append(structures[random_seq][i])
                except:
                    if len(arg_list)<3:
                        arg_list = [1, 4, 5, 6, 8, 9, 9]
                    if 4 not in arg_list:
                        arg_list[0] = 4
        label_structure = generate_label_structure(arg_list)
        spacy_data = generate_spacy_data(label_structure)
        training_data.append(spacy_data)
    return training_data


def training_data_to_json(training_data, output_file):
    classes = set()
    for _, annotation in training_data:
        for _, _, label in annotation['entities']:
            classes.add(label)
            
    classes = sorted(list(classes))

    annotations = []
    for text, annotation in training_data:
        entities = annotation['entities']
        entities_list = [list(entity) for entity in entities]
        annotations.append([text, {"entities": entities_list}])

    json_data = {
        "classes": classes,
        "annotations": annotations
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        

print("Started")
training_data = generate_training_data(100)
training_data_to_json(training_data, "100_train_set.json")
print("ALMOST THERE")
training_data = generate_training_data(20)
training_data_to_json(training_data, "100_train_set_val.json")
# print("Data generation complete.")