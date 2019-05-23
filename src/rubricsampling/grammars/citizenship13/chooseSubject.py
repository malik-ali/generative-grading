import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class ChooseSubject(Decision):
    def registerChoices(self):
        self.addChoice('subject', {
             'they' : 100,
             'the original colonists' : 20,
             'the colonists':50,
             'the pilgrims':5,
             'the english':1
         })

    def renderCode(self):
         value = self.getChoice('subject')
         self.addOrSetState('subject', value)
         return ''
