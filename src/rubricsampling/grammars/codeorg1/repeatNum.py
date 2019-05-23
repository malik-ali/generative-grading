import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class RepeatNum(ReusableDecision):
    def registerChoices(self):
        choice_name = self.getKey()
        self.addChoice(choice_name, {
            '3' : 100,
            '1' : 20,
            '2' : 20,
            'random' : 20,
        })

    def updateRubric(self):
        choice_name = self.getKey()
        choice = self.getChoice(choice_name)
        rubric = {
            '3' : [],
            '1' : ["Doesn't loop three times"],
            '2' : ["Doesn't loop three times"],
            'random' : ["Doesn't loop three times"],
        }
        labelList = rubric[choice]
        for label in labelList:
            self.turnOnRubric(label)
    
    def renderCode(self):
        choice = self.getChoice(self.getKey())
        self.incrementState('repeat_num_count')

        if choice == 'random':
            return str(int(np.random.randint(4, 10)))
        else:
            return choice
