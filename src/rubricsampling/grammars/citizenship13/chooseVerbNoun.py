import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


# this non terminal decides whether our answer is correct
# or not and expands the corresponding nonterminals accordingly
# By the end of this expansion, self.state should have noun and verb populated
class ChooseVerbNoun(Decision):
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
         else:
            # for incorrect answers, default is an incomplete sentence is fine...
            self.addOrSetState('nounAloneOk', True)
            return self.expand('IncorrectAnswer')
