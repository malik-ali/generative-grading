import os
import pickle
import getpass
import numpy as np
from collections import Counter
from src.rubricsampling.generatorUtils import fixWhitespace
from scripts.process_data import strip_comments

N_HEAD = 2
N_TAIL = 30
USER = getpass.getuser()
PROCESSED_DIR = '/mnt/fs5/{}/generative-grading/data/real/education/liftoff/processed'.format(USER)
RAW_DIR = '/mnt/fs5/{}/generative-grading/data/real/education/liftoff/raw'.format(USER)


def remove_imports(p):
    lines = p.split('\n')
    ret = []
    for line in lines:
        if line.strip().startswith('import '):
            continue
        ret.append(line)
    return '\n'.join(ret)


def clean_raw_program(prog):
    # 1. remove extra whitespace
    # 2. remove comments
    # 3. remove import statements
    return remove_imports(strip_comments(fixWhitespace(prog)))


if __name__ == "__main__":
    pickle_path = os.path.join(PROCESSED_DIR, 'raw_student_programs.pkl')
    with open(pickle_path, 'rb') as fp:
        data = pickle.load(fp)

    data2count = Counter()
    for prog in data:
        prog = clean_raw_program(prog)
        data2count[prog] += 1

    head_programs = data2count.most_common(2)
    head_programs = [p[0] for p in head_programs]
    tail_programs = [p[0] for p in data2count.items() if p[1] == 1]
    tail_programs = np.array(tail_programs)
    tail_programs = np.random.choice(tail_programs, size=N_TAIL, replace=False)
    tail_programs = tail_programs.tolist()

    programs = head_programs + tail_programs

    with open('./human_exp_programs.pickle', 'wb') as fp:
        pickle.dump(programs, fp)
