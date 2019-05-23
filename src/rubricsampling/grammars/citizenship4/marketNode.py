import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class MarketNode(Decision):

    def registerChoices(self):
        self.addChoice('marketNoun', {
            'market': 100,
            'economy': 100,
            'market economy': 100,
            'market system': 100,
            'free market': 100,
            'free market system': 100,
            'free market economy': 100,
            'fair': 100,
            'fair market': 100,
            'system': 100,
            'free': 100,
            'regulated': 50,
            'regulated market': 50,
            'regulated free market': 50,
            'regulated market economy': 50,
            'mixed': 50,
            'mixed market': 50,
            'mixed market economy': 50,
            'laissez faire': 50,
            'laissez faire market': 50,
            'laissez-faire': 25,
            'laissez-faire market': 25,
            'laissez': 25,
            'faire': 25,
            'laize faire': 25,
            'laize faire market': 25,
            'dog-eat-dog market': 10,
        })

    def renderCode(self):
        self.addState('noun', self.getChoice('marketNoun'))
        return ''

