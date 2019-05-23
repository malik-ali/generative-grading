import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class ClockwiseTurn(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            'left' : 50,
            'right' : 50,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            'left' : ['Left/Right confusion'],
            'right' : [],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('clockwise_turn_count')
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
                ),
            }
        else:
            raise Exception('how did you get here?')
        
        return gu.format(template, templateVars)
