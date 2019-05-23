import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class MakeSentence(Decision):

    def registerChoices(self):
        self.addChoice('sentenceStyle', {
            'hasPeriod' : 25,
            'hasQuestion' : 25,
            'hasQuotes' : 10,
            'hasNone': 100,
        })
        self.addChoice('sentencePrefix', {
            'noPrefix': 100,
            'thePrefix': 25,
            'aPrefix': 25,
        })

    def renderCode(self):
        if self.getChoice('sentenceStyle') == 'hasNone':
            sentence = '{}'.format(self.getState('noun'))
        elif self.getChoice('sentenceStyle') == 'hasPeriod':
            sentence = '{}.'.format(self.getState('noun'))
        elif self.getChoice('sentenceStyle') == 'hasQuestion':
            sentence = '{}?'.format(self.getState('noun'))
        elif self.getChoice('sentenceStyle') == 'hasQuotes':
            sentence = '"{}"'.format(self.getState('noun'))
        else:
            raise Exception('how did you get here?')

        if self.getChoice('sentencePrefix') == 'noPrefix':
            sentence = '{}'.format(sentence)
        elif self.getChoice('sentencePrefix') == 'thePrefix':
            sentence = 'the {}'.format(sentence)
        elif self.getChoice('sentencePrefix') == 'aPrefix':
            sentence = 'a {}'.format(sentence)
        else:
            raise Exception('how did you get here?')

        return sentence
