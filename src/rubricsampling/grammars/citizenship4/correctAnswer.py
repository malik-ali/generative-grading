import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class CorrectAnswer(Decision):

    def registerChoices(self):
        self.addChoice('correctReason', {
            'market' : 100,
            'capitalism' : 100,
        })

    def renderCode(self):
        if self.getChoice('correctReason') == 'market':
            return '{MarketNode}'
        else:
            return '{CapitalismNode}'
