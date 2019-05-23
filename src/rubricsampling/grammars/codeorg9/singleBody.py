import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class SingleBody(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'correct': 100,
            'incorrect': 15,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'correct': ['Correct body order'],
            'incorrect': ['Single shape: wrong MT order'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('single_body_count')

        if choice == 'correct':
            # assume move and turn do not affect each other
            move_count = self.getState('move_count')
            turn_count = self.getState('turn_count')

            template = '''
            {Move}\n
            {Turn}\n
            '''
            templateVars = {
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(move_count)},
                ),
                'Turn': self.expand(
                    'Turn',
                    {'id': 'turn_{}'.format(turn_count)},
                ),
            }
        elif choice == 'incorrect':
            # assume move and turn do not affect each other
            move_count = self.getState('move_count')
            turn_count = self.getState('turn_count')

            template = '''
            {Turn}\n
            {Move}\n
            '''
            templateVars = {
                'Move': self.expand(
                    'Move',
                    {'id': 'move_{}'.format(move_count)},
                ),
                'Turn': self.expand(
                    'Turn',
                    {'id': 'turn_{}'.format(turn_count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
