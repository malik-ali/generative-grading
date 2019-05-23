import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class Random(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'extraLoop': 1,
            'body': 100,
            'doubleRandom': 1,
        })
    
    def renderCode(self):
        _id = int(self.getKey().split('_')[-1])
        choice = self.getChoice(self.getKey())
        self.incrementState('random_count')

        if choice == 'extraLoop':
            count = self.getState('extra_loop_count')
            template = '''
            {ExtraLoop}
            '''
            templateVars = {
                'ExtraLoop': self.expand(
                    'ExtraLoop',
                    {'id': 'extra_loop_{}'.format(count)},
                ),
            }
        elif choice == 'body':
            count = self.getState('body_count')
            template = '''
            {Body}
            '''
            templateVars = {
                'Body': self.expand(
                    'Body',
                    {'id': 'body_{}'.format(count)},
                ),
            }
        elif choice == 'doubleRandom':
            random0Expansion = self.expand(
                'Random',
                {'id': 'random_{}'.format(_id+1)},
            )
            count = self.getState('random_count')
            random1Expansion = self.expand(
                'Random',
                {'id': 'random_{}'.format(count)},
            )

            template = '''
            {Random0}
            {Random1}
            '''
            templateVars = {
                'Random0': random0Expansion,
                'Random1': random1Expansion,
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
