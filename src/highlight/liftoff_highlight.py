import os
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from src.datasets.student_programs import StudentPrograms
from src.rubricsampling.engineHighlight import InferenceNNHighlight
from src.rubricsampling.generatorUtils import fixWhitespace
from scripts.process_data import strip_comments
from src.utils.io_utils import save_json

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('results_dir', type=str, help='where to save results')
    args = arg_parser.parse_args()

    MODEL_DIR = 'experiments/liftoff_final_100k_tempered/2019-04-20--13_57_14'
    GRAMMAR_DIR = 'src/rubricsampling/grammars/liftoff_hacks'

    student_data = StudentPrograms('liftoff', include_anonymized=False)   
    dataloader = DataLoader(student_data, batch_size=1, shuffle=False)
    tqdm_batch = tqdm(dataloader, total=len(student_data))

    inf_nn = InferenceNNHighlight(GRAMMAR_DIR, MODEL_DIR)

    results = []
    for i, (seq_tokens, seq_lengths, program) in enumerate(tqdm_batch):
        program_args = (seq_tokens, seq_lengths)
        try:
            program, highlights, decisions = inf_nn.guided_sample(program_args)
        except:
            continue
        results.append((program, highlights))
    
    save_json(results, os.path.join(args.results_dir, 'student_highlights.json'))
    
