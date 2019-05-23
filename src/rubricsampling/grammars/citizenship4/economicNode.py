import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class EconomicNode(Decision):

    def registerChoices(self):
        self.addChoice('economicNode', {
            'trade': 100,
            'china': 100,
            'stock market': 100,
            'stocks': 100,
            'businesses': 100,
            'banks': 100,
            'irs': 100,
            'consumerism': 100,
            'marketing': 100,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('economicNode'))
        return ''

