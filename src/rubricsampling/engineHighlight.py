import os
import re
import torch
import pickle
import numpy as np
from copy import deepcopy
from string import Formatter
import torch.nn.functional as F

import generatorUtils as gu
from engine import Engine
import generatorUtils as gu

from src.utils.setup import load_config
from src.agents.autoregressive_rnn import AutoregressiveRNN
from src.rubricsampling.evaluation.base_nn import ProgramNN
from src.utils.io_utils import save_json

from pprint import pprint

def get_formats(string):
    return [t[1] for t in Formatter().parse(string) if t[1] is not None]


def clean_template(template):
    formats = get_formats(template)
    # for form in formats:
        # template = template.replace('{' + form + '}', '(.*)')
    return template 


class EngineHighlight(Engine):

    def render(self, nonterminal, params = {}):
        curr = self._nonterminals[nonterminal]
        curr._setParams(params)

        curr.registerChoices()
        curr.updateRubric()

        render, templates, nodes, extra_highlights = curr.renderCode()
        templates = [clean_template(t) for t in templates]
        to_generate = get_formats(render)

        if not isinstance(templates, list):
            templates = [templates]
        
        template = templates[0] if len(templates) > 0 else render
        highlight_tree = dict(name=nonterminal, template=template, rvs=nodes, children=extra_highlights)

        formatter = dict()
        for format_key in to_generate:
            symbol_to_gen = self.symbol_from_key(format_key)
            formatter[format_key], child_tree = \
                self.render(symbol_to_gen)
            highlight_tree['children'].append(child_tree)
        curr._setParams({}) # clear params

        return render.format(**formatter), highlight_tree

    def renderProgram(self):
        self.reset()
        program, highlights = self.render('Program')
        program = gu.fixWhitespace(program)
        return program, highlights


class EngineGuidedHighlight(EngineHighlight):

    def __init__(self, grammar_dir, exp_dir, strategy='map'):
        super().__init__(grammar_dir)
        assert strategy in ['sample', 'map']
        self.strategy = strategy
        self.exp_dir = exp_dir
        self._load_model(exp_dir)
        self.dataset = self.agent.val_dataset
        self.all_rvs = self.dataset.rv_info['values']

    def _load_model(self, exp_dir):
        config = load_config(os.path.join(exp_dir, 'config.json'))
        self.agent = AutoregressiveRNN(config)
        self.agent.load_checkpoint('checkpoint.pth.tar')
        self.model = self.agent.model
        self.model.eval()
        self.config = config

    def _choicename_to_idx(self, choice_name):
        return self.dataset.rv_info['w2i'][choice_name]

    def _choicename_in_model(self, choice_name):
        return choice_name in self.dataset.rv_info['w2i']

    def _preds_to_value(self, node_ip1, preds, strategy='map'):
        if strategy == 'map':
            idx = np.argmax(preds)
        elif strategy == 'sample':
            idx = np.random.choice(len(preds), p=preds)
        else:
            raise ValueError('Invalid rv value picking strategy')

        rv_name = self.dataset.rv_info['i2w'][str(node_ip1)]
        vals = self.all_rvs[rv_name][0]
        return vals[idx]

    def _rnn_step(self, node_ip1):
        rvOrders_lengths = [torch.from_numpy(np.array(len(self.hidden_store))).to(self.agent.device)]
        node_ip1 = torch.from_numpy(np.array([node_ip1])).to(self.agent.device)
        with torch.no_grad():
            output_i, h0, alphas_i = self.model.step(
                self.node_i, node_ip1, self.program_emb, self.h0, self.hidden_store, 
                self.rvAssignments.long().to(self.agent.device), rvOrders_lengths)
            next_preds = F.softmax(output_i[0], dim=0).cpu().numpy()
            self.h0 = h0
        return next_preds

    def reset(self):
        self.state = {}
        self.choices = {}
        self.rubric = {}

        self.renderOrder = []
        self.rvChoiceOrder = []
        self.preds = []

    def addGlobalChoice(self, choice_name, val):
        if choice_name in self.choices:
            raise ValueError('Key [{}] already in global choices'.format(choice_name))

        self.renderOrder.append(choice_name)
        self.choices[choice_name] = val

        if self._choicename_in_model(choice_name):
            node_idx = self._choicename_to_idx(choice_name)
            val_idx = self.all_rvs[choice_name][0].index(val)
            self.rvAssignments[0][node_idx] = val_idx
            self.node_i = torch.from_numpy(np.array([node_idx])).to(self.agent.device)

    def _pick_rv(self, choice_name, values):
        self.rvChoiceOrder.append(choice_name)
        if not self._choicename_in_model(choice_name) or self.node_i is None:
            val = super(EngineGuidedHighlight, self)._pick_rv(choice_name, values)
            
            tuples = [(v, p) for v, p in values.items()]
            # unpack list of pairs to pair of lists
            choices, ps = list(zip(*tuples))
            ps /= np.sum(ps)            
            display_preds = list(zip(values.keys(), [round(p, 2) for p in ps]))
            return val
        else:
            node_ip1 = self._choicename_to_idx(choice_name)
            preds = self._rnn_step(node_ip1)
            self.preds.append(preds)

            if len(preds) != len(values):
                raise ValueError('pred dim [{len(preds)}] != num values [{len(values)}]')

            val = self._preds_to_value(node_ip1, preds, strategy=self.strategy)
            display_preds = list(zip(values.keys(), [round(p, 2) for p in preds]))
            return val

    def renderProgram(self, program_args, rvAssignments):
        self.reset()

        self.h0 = self.model.init_rnn_hiddens(1)
        self.hidden_store = []

        program_args = [p.to(self.agent.device) for p in program_args]
        self.program_emb = self.model.program_encoder(
            *program_args, return_hiddens=True)
        self.rvAssignments = rvAssignments.clone()
        self.node_i = None

        program, highlights = self.render('Program')
        program = gu.fixWhitespace(program)
        rubricItems = self.getRubricItems()

        return (program, highlights, rubricItems, self.choices, 
                self.rvChoiceOrder, self.rvAssignments)


class InferenceNNHighlight(ProgramNN):

    def __init__(self, grammar_dir, exp_dir, top_k=1, max_iter=50, strategy='sample'):
        self.inf_e = EngineGuidedHighlight(grammar_dir, exp_dir, strategy=strategy)
        self.max_iter = 500
        self.top_k = top_k
        self.strategy = strategy

    def getModelConfig(self):
        return self.inf_e.config

    def guided_sample(self, program_args):
        # something that will crash if accessed without setting
        initAssignments = 1000000 * torch.ones(1, self.inf_e.model.num_nodes)
        program, highlights, labels, decisions, rvOrder, rvAssignments_pred = \
            self.inf_e.renderProgram(program_args, initAssignments)

        return program, highlights, decisions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1)
    args = parser.parse_args() 

    data = []
    e = EngineHighlight('src/rubricsampling/grammars/liftoff_hacks')
    for i in range(args.N):
        program, highlights = e.renderProgram()
        print(program)
        print(highlights)
        print('----')
        print()
        data.append((program, highlights))

    save_json(data, os.path.join('in_grammar_highlights.json'))
    
