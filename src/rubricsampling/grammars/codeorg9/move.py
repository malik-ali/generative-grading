import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class Move(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'correct' : 100,
            'incorrect' : 3,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'correct' : [],
            'incorrect' : ['Move: forward/backward confusion'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        count = self.getState('move_amount_count')
        self.incrementState('move_count')

        if choice == 'correct':
            template = '''
            Move({MoveAmount})\n
            '''
            templateVars = {
                'MoveAmount': self.expand(
                    'MoveAmount',
                    {'id': 'move_amount_{}'.format(count)},
                ),
            }
        elif choice == 'incorrect':
            template = '''
            MoveBackwards({MoveAmount})\n
            '''
            templateVars = {
                'MoveAmount': self.expand(
                    'MoveAmount',
                    {'id': 'move_amount_{}'.format(count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)


class MoveAmount(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'i*10': 100,
            '10*i': 10,
            'i': 20,
            '360/i': 10,
            'iMinusRandom': 3,
            'iDivideRandom': 3,
            'iTimesRandom': 20,
            'randomTimesI': 2,
            '100': 50,
            '10': 50,
            'randomTimesRandom': 50,
            'random': 20,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'i*10': [],
            '10*i': [],
            'i': ['Move: missing opp'],
            '360/i': ['Move: wrong opp'],
            'iMinusRandom': ['Move: wrong opp'],
            'iDivideRandom': ['Move: wrong opp'],
            'iTimesRandom': ['Move: wrong multiple'],
            'randomTimesI': ['Move: wrong multiple'],
            '100': ['Move: constant'],
            '10': ['Move: constant'],
            'randomTimesRandom': ['Move: constant'],
            'random': ['Move: constant'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('move_amount_count')

        if choice == 'iMinusRandom':
            return 'i-{}'.format(np.random.randint(1, 360))
        elif choice == 'iDivideRandom':
            return 'i/{}'.format(np.random.randint(1, 360))
        elif choice == 'iTimesRandom':
            return 'i*{}'.format(np.random.randint(1, 360))
        elif choice == 'randomTimesI':
            return '{}*i'.format(np.random.randint(1, 360))
        elif choice == 'randomTimesRandom':
            rv1 = np.random.randint(1, 360)
            rv2 = np.random.randint(1, 360)
            rv = rv1 * rv2
            return '{}'.format(rv)
        elif choice == 'random':
            return '{}'.format(np.random.randint(1, 360))
        else:
            return choice

