import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class Turn(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'incorrect' : 20,
            'correct' : 80,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'incorrect' : ['Turn: left/right confusion'],
            'correct' : [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        count = self.getState('turn_amount_count')
        self.incrementState('turn_count')

        if choice == 'incorrect':
            template = '''
            TurnLeft({TurnAmount})\n
            '''
            templateVars = {
                'TurnAmount': self.expand(
                    'TurnAmount',
                    {'id': 'turn_amount_{}'.format(count)},
                ),
            }
        elif choice == 'correct':
            template = '''
            TurnRight({TurnAmount})\n
            '''
            templateVars = {
                'TurnAmount': self.expand(
                    'TurnAmount',
                    {'id': 'turn_amount_{}'.format(count)},
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)


class TurnAmount(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            '360/i' : 100,
            'i/360' : 3,
            'iOverRandom' : 3,
            '180/i' : 3,
            '10/i' : 3,
            'randomOverI' : 3,
            'i' : 10,
            'i*10' : 3,
            '360*i' : 3,
            '360-i' : 3,
            '360+i' : 3,
            'i+10' : 3,
            'i-10' : 3,
            'i/10' : 3,
            '120' : 5,
            '72' : 3,
            '40' : 3,
            '60' : 3,
            '90' : 3,
            'randomOverRandom' : 5,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            '360/i' : [],
            'i/360' : ['Turn: wrong opp'],
            'iOverRandom' : ['Turn: wrong opp'],
            '180/i' : ['Turn: wrong multiple'],
            '10/i' : ['Turn: wrong multiple'],
            'randomOverI' : ['Turn: wrong multiple'],
            'i' : ['Turn: missing opp'],
            'i*10' : ['Turn: wrong opp'],
            '360*i' : ['Turn: wrong opp'],
            '360-i' : ['Turn: wrong opp'],
            '360+i' : ['Turn: wrong opp'],
            'i+10' : ['Turn: wrong opp'],
            'i-10' : ['Turn: wrong opp'],
            'i/10' : ['Turn: wrong opp'],
            '120' : ['Turn: constant'],
            '72' : ['Turn: constant'],
            '40' : ['Turn: constant'],
            '60' : ['Turn: constant'],
            '90' : ['Turn: constant'],
            'randomOverRandom' : ['Turn: constant'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('turn_amount_count')

        if choice == 'iOverRandom':
            return 'i/{}'.format(np.random.randint(1, 360))
        elif choice == 'randomOverI':
            return '{}/i'.format(np.random.randint(1, 359))
        elif choice == 'randomOverRandom':
            rv1 = np.random.randint(1, 360)
            rv2 = np.random.randint(1, 360)
            rv = rv1 / float(rv2)
            return '{}'.format(rv)
        else:
            return choice
