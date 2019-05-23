import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class Turn(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'left' : 90,
            'right' : 10,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            'left' : [],
            'right' : ['Left/Right confusion'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('turn_count')

        turn_amount_count = self.getState('turn_amount_count')

        if choice == 'left':
            template = '''
            TurnLeft({TurnAmount}) \n
            '''
            templateVars = {
                'TurnAmount': self.expand(
                    'TurnAmount', 
                    {'id': 'turn_amount_{}'.format(turn_amount_count)}
                ), 
            }
        elif choice == 'right':
            template = '''
            TurnRight({TurnAmount}) \n
            '''
            templateVars = {
                'TurnAmount': self.expand(
                    'TurnAmount', 
                    {'id': 'turn_amount_{}'.format(turn_amount_count)}
                )
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)


class TurnAmount(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            '120' : 100,
            '60' : 50,
            '90' : 50,
            '45' : 30,
            '30' : 30,
            'randomSmall' : 10,
            'randomBig' : 10,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            '120': [],
            '60': ['Does not invert angle'],
            '90': ['Default turn'],
            '45': ['Does not invert angle', 'Does not know equilateral is 60'],
            '30': ['Does not invert angle', 'Does not know equilateral is 60'],
            'randomSmall': ['Does not invert angle', 'Does not know equilateral is 60'],
            'randomBig': ['Does not know equilateral is 60'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)

    def renderCode(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)

        self.incrementState('turn_amount_count')

        if choice == 'randomSmall':
            choice = str(int(np.random.randint(0, 89)))
        elif choice == 'randomBig':
            choice = str(int(np.random.randint(91, 360)))
      
        return choice
