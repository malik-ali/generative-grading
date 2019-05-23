# Script to evaluate and compare different nn strategies

from tqdm import tqdm
from torch.utils.data import DataLoader

import os
import difflib
from collections import Counter

from src.datasets.student_programs import StudentPrograms
import src.utils.paths as paths

from src.rubricsampling.evaluation.inference_nn import InferenceNN
from src.rubricsampling.evaluation.lsh_nn import LshNN
from src.rubricsampling.generatorUtils import fixWhitespace
from scripts.process_data import strip_comments

import numpy as np

from src.utils.io_utils import save_json

import editdistance

# TODO: there is a big annoying bug here with problem name and exp name
def jacc_sim(p1, p2):
    s1 = Counter(p1.split())
    s2 = Counter(p2.split())
    num = 0
    denom = 0
    for k in (s1.keys() & s2.keys()):
        num += min(s1[k], s2[k])

    for k in (s1.keys() | s2.keys()):
        denom += max(s1[k], s2[k])

    return num/denom

def edit_sim(p1, p2):
    return -editdistance.eval(p1, p2)

class NNRecoveryEval:
    SIM_METRICS = {
        'jaccard': jacc_sim,
        'edit_sim': edit_sim
    }

    def __init__(self, problem, raw_rubricsamples_dir, exp_dir, results_dir, top_k=1):
        self.results_dir = results_dir
        self.top_k = top_k

        # need anonymised for lsh to work
        self.student_data = StudentPrograms(problem, include_anonymized=True)   
        self.dataloader = DataLoader(self.student_data, batch_size=1, shuffle=False)

        grammar_dir = paths.grammar_path(problem)
        self.inf_nn = InferenceNN(grammar_dir, exp_dir, top_k=top_k)
        self.lsh_nn = LshNN(raw_rubricsamples_dir, top_k=top_k)

    def transformRawProgram(self, prog):
        def remove_imports(p):
            lines = p.split('\n')
            ret = []
            for line in lines:
                if line.strip().startswith('import '):
                    continue
                ret.append(line)
            return '\n'.join(ret)

        return remove_imports(strip_comments(fixWhitespace(prog)))

    def getNNs(self, raw_program, anon_program, program_args):
        top_k_lsh, top_k_lsh_anon = self.lsh_nn.findNearestNeighbours(anon_program)  # TODO: make raw vs anon customisable?
        top_k_inf = self.inf_nn.findNearestNeighbours(raw_program, program_args=program_args)

        return top_k_lsh, top_k_inf
        #return top_k_lsh_anon, top_k_inf

    def computeDistances(self, distance_data, program, top_k_lsh, top_k_inf):
        for metric_name, sim_metric_fn in self.SIM_METRICS.items():
            if metric_name not in distance_data:
                distance_data[metric_name] = dict(lsh=[], inf=[], lsh_max_idxs=[], inf_max_idxs=[])

            dists_lsh  = [sim_metric_fn(program, p) for p in top_k_lsh]
            dists_inf  = [sim_metric_fn(program, p) for p in top_k_inf]

            max_idx_lsh = int(np.argmax(dists_lsh))
            max_idx_inf = int(np.argmax(dists_inf))

            distance_data[metric_name]['lsh_max_idxs'].append(max_idx_lsh)
            distance_data[metric_name]['inf_max_idxs'].append(max_idx_inf)

            distance_data[metric_name]['lsh'].append(dists_lsh[max_idx_lsh])
            distance_data[metric_name]['inf'].append(dists_inf[max_idx_inf])


    def compareNNs(self):
        tqdm_batch = tqdm(self.dataloader, total=len(self.student_data))

        metadata = dict(top_k=self.top_k)
        distance_data = dict(metadata=metadata)

        for i, data_list in enumerate(tqdm_batch):
            # TODO: this indexing is super hardcoded 
            program_args, program, anon_program = data_list[:2], data_list[-2], data_list[-1]	# for non-anon, non-char model
            #program_args, program, anon_program = data_list[:4], data_list[-2], data_list[-1]	# for anon, not_char model

            # TODO: hack to make it work right now. Investigate why this is happening
            if program_args[1].item() == 0:
                print('SKIPPING STUDENT PROG OF LENGTH 0')
                continue

            # access 0 index to unbatch the batch of size 1
            program = program[0]
            anon_program = anon_program[0]
            clean_prog = self.transformRawProgram(program)

            
            top_k_lsh, top_k_inf = self.getNNs(program, anon_program, program_args)

            if len(top_k_lsh) == 0:
                print(f'SKIPPING programs with 0 LSH')
                print(clean_prog)
                continue

            if len(top_k_lsh) != self.top_k or len(top_k_inf) != self.top_k:
                print(f'[WARNING] Found programs with not enough top_k. Expected {self.top_k}, received (lsh={len(top_k_lsh)}, inf={len(top_k_inf)})')

            self.computeDistances(distance_data, clean_prog, top_k_lsh, top_k_inf)

        save_json(distance_data, os.path.join(self.results_dir, f'recovery_similarity_k_{self.top_k}.json'))


    def printDiff(self, p1, p2):
        lines1 = p1.splitlines()
        lines2 = p2.splitlines()
        for line in difflib.unified_diff(lines1, lines2):
            print(line)
    
    def getBestProgram(self, program, top_k, sim_fn=edit_sim):
        sims  = [sim_fn(program, p) for p in top_k]
        max_idx = int(np.argmax(sims))
        best_program = top_k[max_idx]

        return best_program, sims[max_idx]

    def nnDiffs(self):
        #tqdm_batch = tqdm(self.dataloader, total=len(self.student_data))
        tqdm_batch = self.dataloader
        
        nn_map = dict()

        for i, data_list in enumerate(tqdm_batch):
            # TODO: this indexing is super hardcoded 
            program_args, program, anon_program = data_list[:2], data_list[-2], data_list[-1]	# for non-anon, non-char model
            #program_args, program, anon_program = data_list[:4], data_list[-2], data_list[-1]	# for anon, not_char model

            # TODO: hack to make it work right now. Investigate why this is happening
            if program_args[1].item() == 0:
                print('SKIPPING STUDENT PROG OF LENGTH 0')
                continue

            # access 0 index to unbatch the batch of size 1
            program = program[0]
            anon_program = anon_program[0]
            clean_prog = self.transformRawProgram(program)
            
            top_k_lsh, top_k_inf = self.getNNs(program, anon_program, program_args)

            if len(top_k_lsh) == 0:
                print(f'SKIPPING programs with 0 LSH')
                print(clean_prog)
                continue

            if len(top_k_lsh) != self.top_k or len(top_k_inf) != self.top_k:
                print(f'[WARNING] Found programs with not enough top_k. Expected {self.top_k}, received (lsh={len(top_k_lsh)}, inf={len(top_k_inf)})')

            best_prog_lsh, lsh_edit_sim = self.getBestProgram(clean_prog, top_k_lsh)
            best_prog_inf, inf_edit_sim = self.getBestProgram(clean_prog, top_k_inf)

            nn_map[program] = dict(lsh=best_prog_lsh, inf=best_prog_inf)

            print('=================== Original ===================')
            print(clean_prog)
            #print(anon_program)

            print('===================  LSH ===================')
            print(best_prog_lsh)

            print('=================== Inference ===================')
            print(best_prog_inf)

            print('=================== LSH Diff ===================')
            self.printDiff(clean_prog, top_k_lsh[0])

            print('=================== Inf. Diff ===================')
            self.printDiff(clean_prog, top_k_inf[0])
            
            print()
            print(f"Edit similarity: lsh={lsh_edit_sim}, inf={inf_edit_sim}")

            #input('Enter to continue')
            print()
            print('*'*80)
            print()
        
        #save_json(nn_map, os.path.join(self.results_dir, f'top_{self.top_k}_nns.json'))


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'problem',
        metavar='problem',
        default='None',
        help='The name of the problem to evaluate e.g. liftoff')
    arg_parser.add_argument(
        'raw_rubricsamples_dir',
        type=str,
        help='directory of raw rubric samples to use for LSH')
    arg_parser.add_argument(
        'exp_dir',
        type=str,
        help='directory of model experiment to use for evaluation')
    arg_parser.add_argument(
        'results_dir',
        type=str,
        help='where to save results')
    arg_parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        help='Algorithms will pick best nn out of top_k matches')

    
    args = arg_parser.parse_args()

    # grammar_dir = paths.grammar_path(problem)
    if not 0 < args.top_k <= 10:
        raise ValueError('top_k must be greater 0 and leq 10')

    agent = NNRecoveryEval(args.problem, args.raw_rubricsamples_dir, args.exp_dir, args.results_dir, top_k=args.top_k)
    
    agent.compareNNs()
    #agent.nnDiffs()


