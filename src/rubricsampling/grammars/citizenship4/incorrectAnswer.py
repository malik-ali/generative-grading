import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class IncorrectAnswer(Decision):

    def registerChoices(self):
        self.addChoice('incorrectReason', {
            'economic_terms' : 100,
            'political_terms' : 100,
            'currency_terms': 100,
            'sassy_terms': 50,
        })

    def renderCode(self):
        if self.getChoice('incorrectReason')  == 'economic_terms':
            return '{EconomicNode}'
        elif self.getChoice('incorrectReason')  == 'political_terms':
            return '{PoliticalNode}'
        elif self.getChoice('incorrectReason')  == 'currency_terms':
            return '{CurrencyNode}'
        elif self.getChoice('incorrectReason')  == 'sassy_terms':
            return '{SassyNode}'
        else:
            raise Exception('how did you get here')
