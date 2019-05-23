import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class PoliticalReason(Decision):

    def registerChoices(self):
        self.addChoice('politicalSyntax', {
            'pursue' : 100,
            'escape' : 100,
            'passive':10
        })

    def renderCode(self):
        syntax = self.getChoice('politicalSyntax')
        if syntax == 'pursue': return '{PursuePolitical}'
        if syntax == 'escape': return '{EscapePolitical}'
        if syntax == 'passive': return '{PassivePolitical}'
        raise Exception('unknown syntax: ' + syntax)

class PursuePolitical(Decision):

    def registerChoices(self):
        self.addChoice('posPolVerbChoice', {
            'gain' : 50,
            'obtain' : 50,
            'practice' : 50,
            'find' : 50,
            'pursue':50,
            'seek':50
        })
        self.addChoice('posPolNounChoice', {
            'freedom': 50,
            'beliefs': 50,
            'liberty': 50,
            'political freedom' : 50,
            'political beliefs' : 25,
        })

    def renderCode(self):
        optTheir = self.expand('OptTheir')
        noun = self.getChoice('posPolNounChoice')
        optUs = self.expand('OptUS')
        
        nounPhrase = optTheir + noun + optUs
        self.addOrSetState('noun', nounPhrase)
        self.addOrSetState('verb', self.getChoice('posPolVerbChoice'))
        self.addOrSetState('nounAloneOk', True)
        return ''


class EscapePolitical(Decision):

    def registerChoices(self):
        self.addChoice('negPolVerbChoice', {
            'escape' : 50,
            'escape from' : 50,
            'avoid' : 50,
            'flee' : 50,
            'flee from' : 50,
        })
        self.addChoice('negPolNounChoice', {
            'political persecution' : 50,
            'political prosectuion' : 25,
            'political oppression' : 50,
            'oppression':20
        })

    def renderCode(self):
        optUk = self.expand('OptUK')
        noun = self.getChoice('negPolNounChoice')
        optTheir = self.expand('OptTheir')
        nounPhrase = optTheir + noun + optUk
        self.addOrSetState('noun', nounPhrase)
        self.addOrSetState('verb', self.getChoice('negPolVerbChoice'))
        self.addOrSetState('nounAloneOk', True)
        return ''

class PassivePolitical(Decision):
    def registerChoices(self):
        self.addChoice('passivePolNoun', {
            'politically persecuted' : 50,
            'politically oppressed' : 25,
        })
    def renderCode(self):
        self.addOrSetState('verb', 'be')
        self.addOrSetState('noun', self.getChoice('passivePolNoun'))
        self.addOrSetState('nounAloneOk', True)
        self.addOrSetState('mustBePast', True)
        return ''
