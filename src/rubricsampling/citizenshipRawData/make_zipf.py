import os
import numpy as np
import pandas as pd
from collections import Counter

df = pd.read_csv('./studentanswers_grades_698.tsv', sep='\t')
df = df[df['Q#'] == 13]
answers = df['answer']
answers = np.asarray(answers)

tracker = Counter()
for answer in answers:
    tracker[answer] += 1

count = np.sort(np.array(list(tracker.values())))[::-1]
rank = np.arange(len(count)) + 1

log_count = np.log(count)
log_rank = np.log(rank)

if not os.path.isdir('statistics'):
    os.makedirs('statistics')

np.save('./statistics/log_count.npy', log_count)
np.save('./statistics/log_rank.npy', log_rank)

