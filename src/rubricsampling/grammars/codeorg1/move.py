import os
import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
INT_DOMAIN = np.load(os.path.join(CUR_DIR, 'integer_domain.npy'))


class Move(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'backward' : 5,
            'forward' : 100,
        })
    
    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            'backward' : ['Backwards/Forwards confusion'],
            'forward' : [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())

        self.incrementState('move_count')

        move_amount_count = self.getState('move_amount_count')

        if choice == 'backward':
            template = '''
            MoveBackwards({MoveAmount}) \n
            '''
            templateVars = {
                'MoveAmount': self.expand(
                    'MoveAmount',
                    {'id': 'move_amount_{}'.format(move_amount_count)}
                ), 
            }
        elif choice == 'forward':
            template = '''
            Move({MoveAmount}) \n
            '''
            templateVars = {
                'MoveAmount': self.expand(
                    'MoveAmount',
                    {'id': 'move_amount_{}'.format(move_amount_count)}
                ), 
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)


class MoveAmount(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            '100' : 50,
            '50' : 100,
            'random' : 10,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            '100' : ['Default move'],
            '50' : [],
            'random' : ['Random move amount'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)

        self.incrementState('move_amount_count')

        if choice == 'random':
            choice = str(float(np.random.choice(INT_DOMAIN)))
      
        return choice
