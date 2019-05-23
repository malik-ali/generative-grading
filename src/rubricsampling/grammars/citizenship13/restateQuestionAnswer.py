import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class RestateQuestionAnswer(Decision):

    def registerChoices(self):
        self.addChoice('restateStyle', {
            # 'rhetorical': 100,
            'definitive': 100
        })

    def renderCode(self):
        if self.getChoice('restateStyle') == 'definitive':
            self.expand('DefinitiveStatement')
        else:
            raise Exception('how did you get here?')
        return ''

    
class DefinitiveStatement(Decision):

    def registerChoices(self):
        self.addChoice('defVerbChoice', {
            'be': 100,
            'colonize': 100,
            'create':100
        })
        self.addChoice('defNounChoice', {
            'a colonist': 100,
            'colonists': 100,
            'a colony':100
        })
    
    def renderCode(self):
        self.addOrSetState('noun', self.getChoice('defNounChoice'))
        self.addOrSetState('verb', self.getChoice('defVerbChoice'))
        return ''
