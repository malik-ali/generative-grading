import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class RepeatNum(ReusableDecision):
    def registerChoices(self):
        self.addChoice(self.getKey(), {
            'i' : 100,
            '(i*2)' : 3,
            '(i*10)' : 3,
            '(i+1)' : 10,
            '3' : 10,
            '(360/i)' : 3,
            'random' : 40,
        })

    def updateRubric(self):
        choice = self.getChoice(self.getKey())
        rubric = {
            'i' : ['Correct repeat num'],
            '(i*2)' : ['Single shape: wrong iter #'],
            '(i*10)' : ['Single shape: wrong iter #'],
            '(i+1)' : ['Single shape: wrong iter #'],
            '3' : ['Single shape: wrong iter #'],
            '(360/i)' : ['Single shape: wrong iter #'],
            'random' : ['Single shape: wrong iter #'],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('repeat_num_count')

        if choice == 'random':
            return str(int(np.random.randint(1, 10)))
        else:
            return choice
