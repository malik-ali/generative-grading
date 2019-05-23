import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class ChooseNoun(Decision):
    def registerChoices(self):
        self.addChoice('correctStrategy', {
             True : 100,
             False : 80
        })

    def updateRubric(self):
        if not self.getChoice('correctStrategy'):
             self.turnOnRubric('incorrectAnswer')

    def renderCode(self):
        if self.getChoice('correctStrategy'):
            return '{CorrectAnswer}'
        return '{IncorrectAnswer}'
