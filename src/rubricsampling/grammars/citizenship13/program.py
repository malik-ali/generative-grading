import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

'''
What is one reason the original colonists came to America?
'''
class Program(Decision):
    def registerChoices(self):
        '''
        We add a dummy RV with only one value 
        to make the RNN code easier.
        '''		
        self.addChoice(self.ROOT_RV_NAME, {
            self.ROOT_RV_VAL: 1
        }) 

    def renderCode(self):
        # choose verb noun and its subsequent nonterminals
        # MUST populate the state dictionary with noun and verb
        # TODO: make this compatible with more than one noun
        # and verb in the case that we have multiple correct answers
        # strung together.
        self.expand('ChooseVerbNoun') 
        self.expand('ChooseSubject')

        subject = self.getState('subject')
        noun = self.getState('noun')
        verb = self.getState('verb')

        sentence = self.expand('MakeSentence', {
            'subject':subject,
            'verb':verb, 
            'noun':noun
        })
        return '{OptQualifier}' + sentence + '{OptPunctuation}'

class OptPunctuation(Decision):
    def registerChoices(self):
        self.addChoice('punctuation', {
            '':100,
            '.':50,
            '?':15
        })

    def renderCode(self):
        return self.getChoice('punctuation')

class OptQualifier(Decision):
    def registerChoices(self):
        self.addChoice('qualifier', {
            '':200,
            'i think ':2,
            'maybe ':1,
            'i learned, ':2
        })

    def renderCode(self):
        qualifier = self.getChoice('qualifier')
        return self.getChoice('qualifier')
