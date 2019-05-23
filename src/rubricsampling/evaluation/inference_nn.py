import torch

from src.rubricsampling.evaluation.base_nn import ProgramNN
from src.rubricsampling.engineGuidedInference import EngineGuided


class InferenceNN(ProgramNN):

    def __init__(self, grammar_dir, exp_dir, top_k=1, max_iter=50, strategy='sample'):
        self.inf_e = EngineGuided(grammar_dir, exp_dir, strategy=strategy)
        self.max_iter = 500
        self.top_k = top_k
        self.strategy = strategy

    def getModelConfig(self):
        return self.inf_e.config

    def findNearestNeighbours(self, studentProgram, **kwargs):
        assert self.strategy == 'sample', \
            'for nearest neighbors, strategy MUST be sample'

        if 'program_args' not in kwargs:
            raise ValueError("Need to pass in program args")

        program_args = kwargs['program_args']

        num_tries = 0
        ret = dict()
        while len(ret) < self.top_k and num_tries < self.max_iter:
            sample, decisions = self.guided_sample(program_args)
            ret[sample] = decisions
            num_tries += 1

        # TODO: return full dict when decisions also wanted
        return list(ret.keys())

    def guided_sample(self, program_args, return_labels=False):
        # something that will crash if accessed without setting
        initAssignments = 1000000 * torch.ones(1, self.inf_e.model.num_nodes)
        program, labels, decisions, rvOrder, rvAssignments_pred = \
            self.inf_e.renderProgram(program_args, initAssignments)

        if return_labels:
            return program, labels, decisions

        return program, decisions
