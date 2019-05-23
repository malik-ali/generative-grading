import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class CapitalismNode(Decision):

    def registerChoices(self):
        self.addChoice('capitalismNoun', {
            'capitalism': 100,
            'capitalist': 100,
            'capitalist system': 100,
            'capitalist market': 100,
            'capitalistisc': 100,
            'capitolism': 50,
            'capitolist': 50,
            'capitolistisc': 100,
            'capital': 25,
            'capitol': 25,
            'regulated': 50,
            'free': 50,
            'socio-capitalist': 10,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('capitalismNoun'))
        return ''

