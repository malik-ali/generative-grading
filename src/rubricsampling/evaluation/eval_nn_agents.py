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
from src.rubricsampling.evaluation.random_nn import RandomNN
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

AGENTS = {
    # 'lsh': None,
    'random' : None,
    # 'inf_std_tmp_unif': 'experiments/liftoff_final_100k/2019-04-14--00_20_23/',
    # 'inf_std':  'experiments/liftoff_final_100k_standard/2019-04-20--13_38_11/',
    # 'inf_tmp':  'experiments/liftoff_final_100k_tempered/2019-04-20--13_57_14/',
    # 'inf_unif': 'experiments/liftoff_final_100k_unif/2019-04-20--14_19_46/',
    # 'inf_std_tmp': 'experiments/liftoff_final_100k_standard_tempered/2019-04-20--15_07_57/',
    # 'inf_std_anon': 'experiments/liftoff_final_100k_anon_std/2019-05-11--16_30_16/',
    # 'inf_tmp_anon': 'experiments/liftoff_final_100k_anon_tmp/2019-05-11--16_31_45/',
    # 'inf_std_charlvl': 'experiments/liftoff_final_100k_charlvl/2019-05-11--15_40_29/',
    # 'inf_tmp_charlvl': 'experiments/liftoff_final_100k_charlvl_tmp/2019-05-11--16_26_37/',
}

SIM_METRICS = {
    'jaccard': jacc_sim,
    'edit_sim': edit_sim
}

class NNRecoveryEval:
    def __init__(self, problem, raw_rubricsamples_dir, results_dir, top_k=1):
        self.results_dir = os.path.join(results_dir, f'k_{top_k}')
        os.makedirs(self.results_dir, exist_ok=True)

        self.top_k = top_k
        self.grammar_dir = paths.grammar_path(problem)

        # need anonymised for lsh to work
        self.student_data = StudentPrograms(problem, include_anonymized=True)   
        self.dataloader = DataLoader(self.student_data, batch_size=1, shuffle=False)
        self.lsh_nn = LshNN(raw_rubricsamples_dir, top_k=top_k)
        self.random_nn = RandomNN(raw_rubricsamples_dir, top_k=top_k)

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


    def computeDistances(self, distance_data, program, top_k):
        for metric_name, sim_metric_fn in SIM_METRICS.items():
            if metric_name not in distance_data:
                distance_data[metric_name] = dict(dists=[], max_idxs=[])

            dists  = [sim_metric_fn(program, p) for p in top_k]
            max_idx = int(np.argmax(dists))

            distance_data[metric_name]['max_idxs'].append(max_idx)
            distance_data[metric_name]['dists'].append(dists[max_idx])

    def getBestProgram(self, program, top_k, sim_fn=edit_sim):
        sims  = [sim_fn(program, p) for p in top_k]
        max_idx = int(np.argmax(sims))
        best_program = top_k[max_idx]

        return best_program, sims[max_idx]

    def evaluateAllAgents(self):
        for agent_name, exp_dir in AGENTS.items():
            if agent_name == 'lsh' or agent_name == 'random':
                inf_nn = None
            else:
                inf_nn = InferenceNN(self.grammar_dir, exp_dir, top_k=self.top_k)
                
            self.evaluateAgent(agent_name, inf_nn)
        

    def evaluateAgent(self, agent_name, inf_nn):
        tqdm_batch = tqdm(self.dataloader, total=len(self.student_data))

        distance_data = dict()
        nn_map = dict()

        for i, data_list in enumerate(tqdm_batch):
            program_args, program, anon_program = data_list[:2], data_list[-2], data_list[-1]	# for non-anon, non-char model
            # program_args, program, anon_program = data_list[:4], data_list[-2], data_list[-1]	# for anon xor char model

            # TODO: hack to make it work right now. Investigate why this is happening
            if program_args[1].item() == 0:
                print('SKIPPING STUDENT PROG OF LENGTH 0')
                continue

            # access 0 index to unbatch the batch of size 1
            program = program[0]
            anon_program = anon_program[0]
            clean_prog = self.transformRawProgram(program)
            
            if agent_name == 'lsh':
                top_k_progs = self.lsh_nn.findNearestNeighbours(anon_program)
            elif agent_name == 'random':
                top_k_progs = self.random_nn.findNearestNeighbours(anon_program) 
            else:
                top_k_progs = inf_nn.findNearestNeighbours(program, program_args=program_args)

            if len(top_k_progs) == 0:
                print(f'SKIPPING programs with 0 top_k for agent={agent_name}')
                print(clean_prog)
                continue

            if len(top_k_progs) != self.top_k:
                print(f'[WARNING] Found programs with not enough top_k. Expected {self.top_k}, received [{agent_name}_len={len(top_k_progs)}]')

            best_prog, _ = self.getBestProgram(clean_prog, top_k_progs)
            nn_map[program] = best_prog
            self.computeDistances(distance_data, clean_prog, top_k_progs)

        all_data = dict(distances=distance_data, nns=nn_map)
        save_json(all_data, os.path.join(self.results_dir, f'{agent_name}.json'))


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
        'results_dir',
        type=str,
        help='where to save results')
    arg_parser.add_argument(
        '--top-k',
        type=int,
        default=1,
        help='Algorithms will pick best nn out of top_k matches')
    
    args = arg_parser.parse_args()

    if not 0 < args.top_k <= 10:
        raise ValueError('top_k must be greater 0 and leq 10')

    agent = NNRecoveryEval(args.problem, args.raw_rubricsamples_dir, args.results_dir, top_k=args.top_k)
    
    agent.evaluateAllAgents()


