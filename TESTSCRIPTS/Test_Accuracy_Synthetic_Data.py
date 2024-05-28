import spacy
import json
import matplotlib.pyplot as plt
import numpy as np

# nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("5k-data-improved-best-transformer-acc")

with open("fake_test_data_with_improvements_v2_20k.json") as f:
    total_correct = 0
    total_entries = 0
    data = json.load(f)
    y = []
    for d in data['annotations']:
        doc = nlp(d[0])
        truth_vals = [(x[2],d[0][x[0]:x[1]]) for x in d[1]['entities']]
        entries = len(truth_vals)
        total_entries = total_entries + entries
        correct = 0
        for ent in doc.ents:
            for index, val in enumerate(truth_vals):
                if ent.label_ in val[0] and ent.text in val[1]:
                    print(f"LABEL {ent.label_}, \nTEXT : {ent.text}")
                    correct += 1
                    total_correct += 1
                    truth_vals[index] = ("","")
        print(f"truth-vals : {truth_vals}, \n FOUND_ents {doc.ents}")
        # for ent in doc.ents:
        #     print(ent.text, ent.label_)
        # print("_____________________________________________________")
        # print(f"Correct : {correct} out of {entries} - {(correct/entries)*100}%")
        y.append((correct/entries)*100)
    x = [x for x in range(len(y))]
    print(f"Total correct : {total_correct} out of {total_entries} - {(total_correct/total_entries)*100}%")
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.axhline(y=(total_correct/total_entries)*100, color='r', linestyle='--')
    plt.text(x=len(y)/2, y=(total_correct/total_entries)*100 + 1, s='Average Correct%', color='red')
    plt.xlabel('Entry Index')
    plt.ylabel('Percentage of Correct Identifications')
    plt.title('Correct Identification Percentage per Entry')
    plt.xticks(x) 
    plt.savefig("test.png")
    plt.show()