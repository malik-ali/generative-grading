import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class OptUK(Decision):
    def registerChoices(self):
        self.addChoice('hasUk', {
            'True': 10,
            'False':50
        })

    def renderCode(self):
        if(self.getChoice('hasUk') == 'False'):
            return ''
        else:
             return ' from {Origin}'

class Origin(Decision):
    def registerChoices(self):
        self.addChoice('originChoice', {
            'england': 25,
            'uk': 10,
            'the uk': 50,
            'the united kingdom':10,
            'the british': 25,
            'britain': 25,
            'great britain': 25,
            'europe': 10,
            'the old world':5,
            'their king and queen':10
        })

    def renderCode(self):
        return self.getChoice('originChoice')
