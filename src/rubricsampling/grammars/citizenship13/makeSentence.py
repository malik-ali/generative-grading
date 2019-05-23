import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision
import mlconjug


conjugator = mlconjug.Conjugator(language='en')

# construct the sentence or phrase from the populated state dictionary
class MakeSentence(Decision):

    def registerChoices(self):
        self.addChoice('answerFormat', {
            'nounOnly': 50,
            'fullSentence':30
        })

    def renderCode(self):
        # sometimes we are going to overwrite having the noun only
        canNounBeAlone = self.checkIfNounCanBeAlone()
        hasVerb = self.checkIfHasVerb()
        ansFormat = self.getChoice('answerFormat')
        if hasVerb and (ansFormat == 'fullSentence' or not canNounBeAlone):
            return self.expand('FullSentence', self.params)
        else:
            return self.params['noun']

    def checkIfNounCanBeAlone(self):
        if not self.hasState('nounAloneOk'):
            return False
        return self.getState('nounAloneOk')

    def checkIfHasVerb(self):
        if not self.hasState('hasVerb'):
            return True
        return self.getState('hasVerb')

class FullSentence(Decision):

    def registerChoices(self):
        self.addChoice('conjugate', {
            'true':60,
            'false':40
        })

    def renderCode(self):
        canConj = self.checkCanConjugate()
        mustBeConj = self.checkMustConjugate()
        if mustBeConj or canConj and self.getChoice('conjugate') == 'true':
            return self.expand('Conjugate', self.params)
        else:
            return self.expand('Infinitive', self.params)

    def checkMustConjugate(self):
        if not self.hasState('mustBePast'): 
            return False
        return self.getState('mustBePast')

    def checkCanConjugate(self):
        if not self.hasState('conjugateOk'): 
            return True
        return self.getState('conjugateOk')

class Conjugate(Decision):
    def registerChoices(self):
        self.addChoice('tense_conj', {
            'indicative present':20,
            'indicative past tense':50,
            'past continuous':20,
            'had':20
        })

    def renderCode(self):
        tense = self.getChoice('tense_conj')

        # some verbs can't be present...
        mustBePast = self.checkMustBePast()
        if tense == 'indicative present' and mustBePast:
            tense = 'past continuous'
            
        subj = self.params['subject']
        verbPhrase = self.params['verb']
        verb = verbPhrase.split(' ')[0]
        verbPost = ' '.join(verbPhrase.split(' ')[1:])
        noun = self.params['noun']

        conj = self.getConjugations(verb)
        if tense.startswith('indicative'):
            conjugated = conj['indicative'][tense]['3p']
        else:
            if tense == 'had':
                conjugated = 'had ' + conj['indicative']['indicative present perfect']['3p']
            if tense == 'past continuous':
                conjugated = 'were ' + conj['indicative']['indicative present continuous']['3p 3p']

        if verbPost != '':
            conjugated += ' ' + verbPost
        return '{} {} {}'.format(subj, conjugated, noun)

    def getConjugations(self, verb):
        # spread is no good...
        if verb == 'spread':
            return {
                'indicative': {
                    'indicative past tense': {'3p':'spread'},
                    'indicative present': {'3p':'spread'},
                    'indicative present perfect': {'3p':'spread'},
                    'indicative present continuous': {'3p 3p':'spreading'},
                }
            }
        return conjugator.conjugate(verb).conjug_info

    def checkMustBePast(self):
        if not self.hasState('mustBePast'): 
            return False
        return self.getState('mustBePast')

class Infinitive(Decision):

    def registerChoices(self):
        self.addChoice('tense_inf', {
            'infinitive':50,
            'wanted':50,
            'came':30,
            'left':30
        })

    def renderCode(self):

        tense = self.getChoice('tense_inf')
        subj = self.params['subject']
        verbPhrase = self.params['verb']
        verb = verbPhrase.split(' ')[0]
        verbPost = ' '.join(verbPhrase.split(' ')[1:])
        noun = self.params['noun']

        if tense == 'infinitive':
            return 'to {} {}'.format(verb, noun)

        # you can optionally prepand had
        if tense == 'wanted':
            optHad = self.expand('OptHad')
            conjugated = optHad + 'wanted to ' + verb
        if tense == 'came':
            conjugated = 'came to ' + verb
        if tense == 'left':
            optHad = self.expand('OptHad')
            conjugated = 'left to ' + verb

        if verbPost != '':
            conjugated += ' ' + verbPost


        return '{} {} {}'.format(subj, conjugated, noun)
    
