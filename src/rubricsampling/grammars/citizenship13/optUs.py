import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class OptUS(Decision):
    def registerChoices(self):
        self.addChoice('hasUs', {
            'True': 5,
            'False':50
        })

    def renderCode(self):
        if(self.getChoice('hasUs') == 'False'):
            return ''
        else:
             return ' in {Destination}'

class Destination(Decision):
    def registerChoices(self):
        self.addChoice('destinationChoice', {
            'the us': 30,
            'america': 30,
            'the new world': 5,
            'the colonies':5,
            'virginia':1,
            'massachusetts':1
        })

    def renderCode(self):
        return self.getChoice('destinationChoice')
