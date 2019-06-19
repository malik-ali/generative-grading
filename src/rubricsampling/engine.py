import os
import sys
import random
from os.path import dirname, basename, isfile
import glob
import inspect
import re
import copy
import numpy as np

from collections import deque, defaultdict
from string import Formatter
import generatorUtils as gu


CUR_DIR = os.path.dirname(__file__)
CODEORG_DIR = os.path.join(CUR_DIR, '..', 'utils', 'codeorg_utils')
CODEORG_DIR = os.path.realpath(CODEORG_DIR)
sys.path.append(CODEORG_DIR)

import blockToTree, treeToString


GRAMMAR = 'liftoff'

class Engine:
    _nonterminalCache = None

    def __init__(self, grammar_dir):
        self.reset()
        self._nonterminals, self._reusableNonterminals = self._getNonTerminals(grammar_dir)

        # we need to do this bc we need special processing to handle this
        self._isCodeOrg = 'codeorg' in grammar_dir

    def reset(self):
        self.state = {}
        self.choices = {}
        self.rubric = {}

    def processCodeOrgProgram(self, program, rubricItems, choices):

        def applyPostFunctions(program, rubricItems, choices, proba=0.05):
            # https://github.com/mhw32/program-vae/blob/master/programvae/tools/run/simulate.py
            choices['doesNotGetNesting'] = False

            if 'Repeat' in program:
                if random.random() < proba:
                    repeatStart = program.index('Repeat')
                    blockStart = program.index('{', repeatStart) + 1
                    blockEnd = program.index('}', repeatStart)
                    body = program[blockStart:blockEnd].strip()

                    # punt on bodies with nesting loops
                    if '{' in body:
                        return program, rubricItems, choices

                    lines = body.split('\n')
                    # punt if num lines is less that or equal to 2
                    if len(lines) < 2:
                        return program, rubricItems, choices

                    # put only the first line in the repeat
                    newBlock = lines[0] +  '\n'
                    newBlock += '}\n'
                    for line in lines[1:]:
                        newBlock += line + '\n'
                    beforeBlock = program[:blockStart]
                    afterBlock = program[blockEnd+1:]

                    program = beforeBlock + '\n' + newBlock + '\n' + afterBlock
                    choices['doesNotGetNesting'] = True

                    return program, rubricItems, choices

            return program, rubricItems, choices

        program = program.replace('Program\n', '')
        # program, rubricItems, choices = applyPostFunctions(
        #     program, rubricItems, choices)
        tree = blockToTree.convert(program)
        tokenList = treeToString.flatten_tree(tree)


        return ' '.join(tokenList), rubricItems, choices, program

    def renderProgram(self):
        while True:
            self.reset()

            try:
                program = self.render('Program')
            except BaseException as e:
                if 'Choice name' in str(e):
                    continue
                else:
                    raise e

            program = gu.fixWhitespace(program)
            rubricItems = self.getRubricItems()
            choices = self.choices

            code = None
            if self._isCodeOrg:
                program, rubricItems, choices, code = self.processCodeOrgProgram(
                    program, rubricItems, choices)

            return program, rubricItems, choices, code

    def getRubricItems(self):
        return self.rubric

    def _pick_rv(self, choice_name, values):
        if type(values) is list:
            # reduce uniform case to dictionary case
            values = {val: 1 for val in values}

        tuples = [(v, p) for v, p in values.items()]
        # unpack list of pairs to pair of lists
        choices, ps = list(zip(*tuples))
        ps /= np.sum(ps)

        # operate on indices so numpy doesnt do weird typecasts
        choice_idx = np.random.choice(range(len(choices)), p=ps)
        return choices[choice_idx]

    def addGlobalChoice(self, choice_name, val):
        if choice_name in self.choices:
            raise ValueError('Key [{}] already in global choices'.format(choice_name))

        self.choices[choice_name] = val

    def _compileRegisteredIds(self, nonterminals, reusableNonterminals):
        '''
         This method should loop through all decisions and get their preregistered ids.
         This will give us an overview of how each reusable decision is intended to be used.

         We group these registered ids by the ReusableDecision they are relevatn to and
         return a dictionary from ReusableDecision name to set of valid ids for that decision.
        '''
        # we keep track of the reusable non terminals

        allRegisteredIds = defaultdict(list)
        for nonterminal in nonterminals.values():
            registeredIds = nonterminal.preregisterDecisionIds()
            if registeredIds is None:   # no ids registered
                continue

            for reusable_decision, valid_ids in registeredIds.items():
                allRegisteredIds[reusable_decision].extend(list(valid_ids))

        registeredButNotReusable = set(allRegisteredIds.keys()) - reusableNonterminals
        reusableButNotRegistered = reusableNonterminals - set(allRegisteredIds.keys())
        # make sure evely registeredId corresponds to a reusable decision.
        if registeredButNotReusable:
            raise ValueError(f'Invalid registration of non-reusable decisions {registeredButNotReusable}')

        # print warnings for reusable decisions that dont have valid ids registered?
        if reusableButNotRegistered:
            print(f'WARNING: reusable decisions not registered anywhere: {reusableButNotRegistered}')
            input('Press enter to continue: ')

        return allRegisteredIds

        # TODO: make it so that any runtime registerChoices is not dynamic---throw error if. Like rubric pipeline

    def _setResusableDecisionIds(self, nonterminals, reusableNonterminals):
        registeredIds  = self._compileRegisteredIds(nonterminals, reusableNonterminals)
        for reusable in reusableNonterminals:
            validIds = registeredIds[reusable]
            nonterminals[reusable].registerValidIds(validIds)

    def _getNonTerminals(self, grammar_dir):
        nonterminals, reusableNonterminals = self._loadNonTerminals(grammar_dir)
        self._setResusableDecisionIds(nonterminals, reusableNonterminals)   # modifies params in place
        return nonterminals, reusableNonterminals

    def _loadNonTerminals(self, grammar_dir):
        # if Engine._nonterminalCache != None:
        #   return Engine._nonterminalCache
        nonterminals = {}
        reusableNonterminals = set()
        file_paths = glob.glob(os.path.join(grammar_dir, "*.py"))
        files = [ f[:-3].replace(os.path.sep, '.') for f in file_paths if isfile(f) and not f.endswith('__init__.py')]
        for f in files:
            module = __import__(f, fromlist=['object'])
            for obj in  dir(module):
                if not obj.startswith('__'):
                    clazz = module.__getattribute__(obj)
                    # TODO: fix hacky conditions

                    if inspect.isclass(clazz) and clazz.__base__.__name__.endswith('Decision') and not clazz.__name__ == 'ReusableDecision':
                        name = clazz.__name__
                        if clazz.__base__.__name__ == 'ReusableDecision':
                            reusableNonterminals.add(name)

                        if name in nonterminals:
                            raise ValueError('Repeated name for nonterminal: {}'.format(name))
                        nonterminals[name] =  clazz(self)
        Engine._nonterminalCache = nonterminals
        return nonterminals, reusableNonterminals

    def symbol_from_key(self, format_key):
        match = re.search(r'([A-Za-z]+)(_\d)?', format_key)
        return match.group(1)

    def render(self, nonterminal, params = {}):
        curr = self._nonterminals[nonterminal]
        curr._setParams(params)

        curr.registerChoices()
        curr.updateRubric()
        render = curr.renderCode()

        to_generate = [t[1] for t in Formatter().parse(render) if t[1] is not None]

        formatter = dict()
        for format_key in to_generate:
            symbol_to_gen = self.symbol_from_key(format_key)
            formatter[format_key] = self.render(symbol_to_gen)
        curr._setParams({}) # clear params

        return render.format(**formatter)


if __name__ == "__main__":
    e = Engine('grammars/'+GRAMMAR)
    for i in range(1000):
        program, rubric, choices, _ = e.renderProgram()
        print(choices)
        print(program)
        print(rubric)
        print('----')
