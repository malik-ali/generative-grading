import os
import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
INT_DOMAIN = np.load(os.path.join(CUR_DIR, 'integer_domain.npy'))


class ClockwiseTurnStart(Decision):
    def registerChoices(self):
        self.addChoice('clockwiseTurnStartChoices', {
            'TurnLeft(60) \n' : 50,
            'TurnLeft(30) \n' : 20,
            'TurnLeft(120) \n' : 100,
            'random' : 10,
        })
    
    def renderCode(self):
        choice = self.getChoice('clockwiseTurnStartChoices')

        if choice == 'random':
            template = '''
            TurnLeft({VeryRandom}) \n
            '''
            templateVars = {
                'VeryRandom': self.expand('VeryRandom'),
            }
            return gu.format(template, templateVars)
        else:
            return choice


class VeryRandom(Decision):
    def renderCode(self):
        return str(float(np.random.choice(INT_DOMAIN)))
