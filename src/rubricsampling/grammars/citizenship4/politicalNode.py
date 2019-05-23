import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class PoliticalNode(Decision):

    def registerChoices(self):
        self.addChoice('politicalNoun', {
            'political party': 100,
            'political': 100,
            'party': 100,
            'democratic party': 100,
            'democratic': 100,
            'democrat': 100,
            'republican party': 100,
            'republican': 100,
            'republic': 100,
            'gop': 100,
            'president': 100,
            'congress': 100,
            'legislative': 100,
            'legislative branch': 100,
            'socialist': 100,
            'communist': 100,
            'socialism': 100,
            'communism': 100,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('politicalNoun'))
        return ''

