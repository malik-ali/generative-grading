import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class SassyNode(Decision):

    def registerChoices(self):
        self.addChoice('sassyNode', {
            'i dont know': 100,
            'i don\'t know': 100,
            'a total failure': 100,
            'not sure': 100,
            'who cares': 100,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('sassyNode'))
        return ''

