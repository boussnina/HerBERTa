import spacy
import json
import matplotlib.pyplot as plt
import numpy as np

with open("efo.json") as f:
    raw_data = json.load(f)
    full_sentence = []
    entry = ""
    for data in raw_data:
        if data.strip() == "EOF":
            if entry:
                full_sentence.append(entry.strip())
            entry = ""
        else:
            entry += data.rstrip().replace("\n", "") + " "
    if entry:
        full_sentence.append(entry.strip())

nlp = spacy.load("5k-toke2vec-acc_with-val")

with open("tags_real_data.json") as f:
    tags = json.load(f)
total_tags = sum([len(x) for x in tags])
total_entries = 0
total_correct = 0
y = []
#Brug NER på alle sætninger
for i, sentence in enumerate(full_sentence):
    print("------------------")
    correct = 0
    tags_in_sentence = len(tags[i])
    total_entries = total_entries + tags_in_sentence
    doc = nlp(sentence)
    #Iterer over alle entities i en sætning
    for ent in doc.ents:
        if [ent.text, ent.label_] in tags[i]:
            print(ent.text, ent.label_, i)
            correct += 1
            total_correct += 1
    y.append((correct/tags_in_sentence)*100)
    # print(f"Correct : {correct} out of {tags_in_sentence} - {(correct/tags_in_sentence)*100}%")

x = [x for x in range(len(full_sentence))]
# print(f"Total correct : {total_correct} out of {total_entries} - {(total_correct/total_entries)*100}%")

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