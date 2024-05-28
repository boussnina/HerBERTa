import spacy
import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_similarity(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    longest = 0

    for i in range(len(str1)):
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                matrix[i+1][j+1] = matrix[i][j] + 1
                longest = max(longest, matrix[i+1][j+1])
            else:
                matrix[i+1][j+1] = 0

    longer_length = max(len(str1), len(str2))
    similarity = longest / longer_length if longer_length > 0 else 0
    return similarity

def merge_entities(entities):
    merged = []
    previous = None
    for entity in entities:
        if previous and entity.label_ == previous.label_ and entity.start == previous.end:
            previous = spacy.tokens.Span(entity.doc, previous.start, entity.end, label=entity.label_)
        else:
            if previous:
                merged.append(previous)
            previous = entity
    if previous:
        merged.append(previous)
    return merged

with open("15ComputerWriting.json") as f:
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

nlp = spacy.load("model-best-5k-Garbage")

with open("15ComputerWriting_tags-garbage.json") as f:
    tags = json.load(f)
specific_tags = ['PLANT', 'LOC', 'PERSON', 'DATE', 'GARBAGE']
tag_stats = {tag: {'total_expected': 0, 'total_identified': 0, 'correct': 0, 'incorrect': 0} for tag in specific_tags}

for tag_list in tags:
    for tag in tag_list:
        if tag[1] in specific_tags:
            tag_stats[tag[1]]['total_expected'] += 1

for i, sentence in enumerate(full_sentence):
    doc = nlp(sentence)
    merged_ents = merge_entities(doc.ents)
    for ent in merged_ents:
        if ent.label_ in specific_tags:
            tag_stats[ent.label_]['total_identified'] += 1
            match_found = False
            for tag in tags[i]:
                if ent.label_ == tag[1] and calculate_similarity(ent.text.lower(), tag[0].lower()) >= 0.7:
                    tag_stats[ent.label_]['correct'] += 1
                    match_found = True
                    break
            if not match_found:
                tag_stats[ent.label_]['incorrect'] += 1

for tag, stats in tag_stats.items():
    print(f"{tag}: Expected {stats['total_expected']}, Identified {stats['total_identified']}, Correct {stats['correct']}, Incorrect {stats['incorrect']}")

categories = list(tag_stats.keys())
correct_percentages = [(stats['correct'] / stats['total_expected'] * 100 if stats['total_expected'] > 0 else 0) for tag, stats in tag_stats.items()]
incorrect_percentages = [(stats['incorrect'] / stats['total_identified'] * 100 if stats['total_identified'] > 0 else 0) for tag, stats in tag_stats.items()]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width/2, correct_percentages, width, label='Correct')
rects2 = ax.bar(x + width/2, incorrect_percentages, width, label='Incorrect')

ax.set_xlabel('Entity Categories')
ax.set_ylabel('Percentage')
ax.set_title('NER Model Performance by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f%%' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
