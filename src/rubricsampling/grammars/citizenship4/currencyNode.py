import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class CurrencyNode(Decision):

    def registerChoices(self):
        self.addChoice('currencyNode', {
            'dollar bill': 100,
            'dollar': 100,
            'money': 100,
            'payment': 100,
            'finance': 100,
            'resource': 100,
            'finances': 100,
            'resources': 100,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('currencyNode'))
        return ''

