import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

'''
Error cases:
they flee freedom to practice religion
to practice freedom to practice religion
they find religious oppression
'''
class CorrectAnswer(Decision):

    def registerChoices(self):
        self.addChoice('reason', {
            'religious' : 100,
            'political' : 100,
            'economicOpportunity':10
        })

    def renderCode(self):
        reason = self.getChoice('reason')
        if reason  == 'religious':
            return '{ReligiousReason}'
        if reason == 'political':
            return '{PoliticalReason}'
        if reason == 'economicOpportunity':
            return '{EconomicOpportunity}'
