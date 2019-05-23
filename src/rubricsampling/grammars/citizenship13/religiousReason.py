import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class ReligiousReason(Decision):

    def registerChoices(self):
        self.addChoice('religiousSyntax', {
            'escape' : 100,
            'pursue' : 100,
            'worship':20,
            'breakAway':3,
            'passive':10 # passive voice
        })


    def renderCode(self):
        syntax = self.getChoice('religiousSyntax')
        if syntax == 'pursue':
            self.expand('PursueReligious')
        elif syntax == 'worship':
            self.expand('WorshipReligious')
        elif syntax == 'breakAway':
            self.expand('BreakAwayReligoius')
        elif syntax == 'passive':
            self.expand('PassiveReligious')
        else:
            self.expand('EscapeReligious')
        return ''

class WorshipReligious(Decision):

    def registerChoices(self):
        self.addChoice('posRelVerbChoice_worship', {
            'worship' : 50,
            'practice':10
        })
        self.addChoice('posRelNounChoice_worship', {
            'their religion': 50,
            'a different religion': 20,
            'any religion': 10,
            'religion':10,
            'freely':10,
            'their religion free from persecution':10
        })

    def renderCode(self):
        self.addOrSetState('noun', self.getChoice('posRelNounChoice_worship'))
        self.addOrSetState('verb', self.getChoice('posRelVerbChoice_worship'))
        self.addOrSetState('conjugateOk', False)
        return ''

# we could consider decomposing this differently into verb and noun choices
# since the verb choices will be reused by the politicalReason nonterminal
class PursueReligious(Decision):

    def registerChoices(self):
        self.addChoice('posRelVerbChoice_pursue', {
            'gain' : 50,
            'obtain' : 50,
            'practice' : 50,
            'find' : 50,
            'pursue':50,
            'seek':50
        })
        self.addChoice('posRelNounChoice_pursue', {
            'freedom': 50,
            'religion': 50,
            'religious freedom' : 50,
            'religious beliefs' : 25,
            'freedom of religion' : 50,
            'freedom to practice religion' : 50,
        })
        self.addChoice('pursueQualifier', {
            '':100,
            'their ':20,
            'more ':20
        })

    def renderCode(self):
        chosenNoun = self.getChoice('posRelNounChoice_pursue')
        optUs = self.expand('OptUS')
        optQualifier = self.getChoice('pursueQualifier')

        nounPhrase = optQualifier + chosenNoun + optUs
        self.addOrSetState('noun', nounPhrase)
        self.addOrSetState('verb', self.getChoice('posRelVerbChoice_pursue'))
        self.addOrSetState('nounAloneOk', True)
        return ''


class EscapeReligious(Decision):

    def registerChoices(self):
        self.addChoice('negRelVerbChoice', {
            'escape' : 50,
            'avoid' : 50,
            'flee' : 50,
            'flee from' : 50,
        })
        self.addChoice('negRelNounChoice', {
            'religious persecution' : 50,
            'religious prosectuion' : 25,
            'religious oppression' : 50,
            'religious tyranny' : 50,
            'tyranny': 25,
        })

    def renderCode(self):
        noun = self.getChoice('negRelNounChoice')
        optTheir = self.expand('OptTheir')
        optUk = self.expand('OptUK')
        nounPhrase = optTheir + noun + optUk
        self.addOrSetState('noun', nounPhrase)
        self.addOrSetState('verb', self.getChoice('negRelVerbChoice'))
        self.addOrSetState('nounAloneOk', True)
        return ''

# todo, consider expanding
class BreakAwayReligoius(Decision):
    def renderCode(self):
        self.addOrSetState('verb', 'break away from')
        self.addOrSetState('noun', 'the church')
        self.addOrSetState('nounAloneOk', False)
        self.addOrSetState('conjugateOk', False)
        return ''

class PassiveReligious(Decision):
    def registerChoices(self):
        self.addChoice('passiveRelNoun', {
            'religiously persecuted' : 50,
            'religiously oppressed' : 25,
        })
    def renderCode(self):
        self.addOrSetState('verb', 'be')
        self.addOrSetState('noun', self.getChoice('passiveRelNoun'))
        self.addOrSetState('nounAloneOk', True)
        self.addOrSetState('mustBePast', True)
        return ''

