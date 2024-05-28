import spacy
import json
import matplotlib.pyplot as plt
import numpy as np
from thefuzz import fuzz

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

with open("linas_100_manual_ocr.json") as f:
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

nlp = spacy.load("2.6k-Trans_with-val")   

with open("linas_100_manual_ocr_tags.json") as f:
    tags = json.load(f)
specific_tags = ['PLANT', 'LAT', 'LON', 'PERSON', 'LOC', 'DATE']
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
                if fuzz.ratio(ent.text.lower(), tag[0].lower()) >= 60:
                    if ent.label_ == tag[1]:
                        tag_stats[ent.label_]['correct'] += 1
                        match_found = True
                        break
            if not match_found:
                tag_stats[ent.label_]['incorrect'] += 1

total_precision = total_recall = total_f1 = 0
total_expected_entities = sum(stats['total_expected'] for stats in tag_stats.values())
for tag, stats in tag_stats.items():
    precision = stats['correct'] / stats['total_identified'] if stats['total_identified'] > 0 else 0
    recall = stats['correct'] / stats['total_expected'] if stats['total_expected'] > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    weight = stats['total_expected'] / total_expected_entities if total_expected_entities > 0 else 0
    weighted_f1 = f1_score * weight
    stats.update({'precision': precision, 'recall': recall, 'f1_score': f1_score, 'weighted_f1': weighted_f1})
    total_precision += precision
    total_recall += recall
    total_f1 += f1_score

average_precision = total_precision / len(tag_stats)
average_recall = total_recall / len(tag_stats)
average_f1 = total_f1 / len(tag_stats)
average_weighted_f1 = sum(stats['weighted_f1'] for stats in tag_stats.values()) / len(tag_stats)

categories = list(tag_stats.keys())
correct_percentages = [stats['correct'] / stats['total_expected'] * 100 if stats['total_expected'] > 0 else 0 for stats in tag_stats.values()]
incorrect_percentages = [stats['incorrect'] / stats['total_identified'] * 100 if stats['total_identified'] > 0 else 0 for stats in tag_stats.values()]

print("\nEntity Performance Summary:")
for tag, stats in tag_stats.items():
    print(f"{tag}: {stats['correct']}/{stats['total_expected']} - Weighted F1: {stats['weighted_f1']:.4f}")

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(categories))
width = 0.35
rects1 = ax.bar(x - width/2, correct_percentages, width, label='Correct')
rects2 = ax.bar(x + width/2, incorrect_percentages, width, label='Incorrect')

ax.set_ylim(0, 110)
ax.set_xlabel('Entity Categories')
ax.set_ylabel('Percentage')
ax.set_title('NER TEXT - transformer (20k) eff with validation')
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

fig, ax = plt.subplots(figsize=(12, 4))
data = [(k, "{:.4f}".format(v['precision']), "{:.4f}".format(v['recall']), "{:.4f}".format(v['f1_score']), "{:.4f}".format(v['weighted_f1'])) for k, v in tag_stats.items()]
data.append(("Average", "{:.4f}".format(average_precision), "{:.4f}".format(average_recall), "{:.4f}".format(average_f1), "{:.4f}".format(average_weighted_f1)))
column_labels = ["Entity", "Precision", "Recall", "F1 Score", "Weighted F1"]

ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=data, colLabels=column_labels, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Scale table to make it more readable
plt.show()
