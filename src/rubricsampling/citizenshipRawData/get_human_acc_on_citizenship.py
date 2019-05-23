import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv('./studentanswers_grades_698.tsv', sep='\t')
df = df[df['Q#'] == 13]
df = df.dropna()
answers = df[['answer', 'G1', 'G2', 'G3']]
answers = np.asarray(answers)

tracker = Counter()
for answer in answers:
    response = answer[0]
    tracker[response.lower()] += 1

zipf_dict = {}
for x, y  in tracker.items():
    if y > 1:
        zipf_dict[x] = 1
    else:
        zipf_dict[x] = 0

correct = []
preds = []
for answer in answers:
    text, g1, g2, g3 = answer
    g1, g2, g3 = int(g1), int(g2), int(g3)
    majority = Counter([g1, g2, g3]).most_common()[0][0]
    correct.append([g1 == majority, g2 == majority, g3 == majority, zipf_dict[text]])
    preds.append([g1, g2, g3, majority])

correct = np.array(correct)
preds = np.array(preds)
f1_1 = f1_score(preds[:, 3], preds[:, 0])
f1_2 = f1_score(preds[:, 3], preds[:, 1])
f1_3 = f1_score(preds[:, 3], preds[:, 2])

prec_1 = precision_score(preds[:, 3], preds[:, 0])
prec_2 = precision_score(preds[:, 3], preds[:, 1])
prec_3 = precision_score(preds[:, 3], preds[:, 2])

recall_1 = recall_score(preds[:, 3], preds[:, 0])
recall_2 = recall_score(preds[:, 3], preds[:, 1])
recall_3 = recall_score(preds[:, 3], preds[:, 2])

_f1_1 = (2 * prec_1 * recall_1) / (prec_1 + recall_1)
_f1_2 = (2 * prec_2 * recall_2) / (prec_2 + recall_2)
_f1_3 = (2 * prec_3 * recall_3) / (prec_3 + recall_3)

print(f1_1, f1_2, f1_3)
print(_f1_1, _f1_2, _f1_3)
print(prec_1, prec_2, prec_3, np.mean([prec_1, prec_2, prec_3]))
print(recall_1, recall_2, recall_3, np.mean([recall_1, recall_2, recall_3]))

