import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


'''
Error casses
----------
to be a a colonist
'''
class IncorrectAnswer(Decision):

  def registerChoices(self):
    self.addChoice('incorrectStyle', {
      'restateQuestion' : 100,
      'gold': 100,
      'land' : 100,
      'tobacco': 100,
      'penalColony':40,
      'explore':100,
      'spreadReligion':100,
      'toGetAway':100,
      'unfairTaxation':100,
      'passive':100
    })

  def renderCode(self):
    style = self.getChoice('incorrectStyle')
    
    if style == 'restateQuestion': return '{RestateQuestionAnswer}'
    if style == 'gold': return '{Gold}' 
    if style == 'land': return '{Land}'
    if style == 'tobacco': return '{Tobacco}'
    if style == 'penalColony': return '{PenalColony}'
    if style == 'explore': return '{Explore}'
    if style == 'spreadReligion': return '{SpreadReligion}'
    if style == 'toGetAway': return '{ToGetAway}'
    if style == 'unfairTaxation': return '{UnfairTaxation}'
    if style == 'passive': return '{IncorrectPassive}'
    else:
      raise Exception('unknown style: '+ style)
      # self.addOrSetState('noun', self.getChoice('incorrectStyle'))
      # self.addOrSetState('verb', 'pursue')
      # return ''


class Gold(Decision):
  def registerChoices(self):
    self.addChoice('goldVerb', {
        'find': 100,
        'make': 100,
    })
    self.addChoice('goldNoun', {
        'gold':100,
        'the gold':20,
        'money': 25
    })

  def renderCode(self):
    self.addOrSetState('verb', self.getChoice('goldVerb'))
    self.addOrSetState('noun', self.getChoice('goldNoun'))
    return ''

class Land(Decision):
  def registerChoices(self):
    self.addChoice('landVerb', {
      '':100,
      'settle':20,
    })

    self.addChoice('landNoun', {
      'land':100,
      'the land':20,
      'land ownership':20
    })

  def renderCode(self):
    
    nounPhrase = self.getChoice('landNoun')
    if self.getChoice('landVerb') == '':
      self.addOrSetState('hasVerb', False)
      nounPhrase = self.expand('OptFor') + nounPhrase
    self.addOrSetState('verb', self.getChoice('landVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class Tobacco(Decision):
  def registerChoices(self):
    self.addChoice('tobaccoVerb', {
      '':100,
      'grow':20,
      'create':10,
    })

    self.addChoice('tobaccoNoun', {
      'tobacco':100,
      'tobacco plantations':20,
      'plantations': 25
    })

  def renderCode(self):
    
    nounPhrase = self.getChoice('tobaccoNoun')
    if self.getChoice('tobaccoVerb') == '':
      self.addOrSetState('hasVerb', False)
      nounPhrase = self.expand('OptFor') + nounPhrase
    self.addOrSetState('verb', self.getChoice('tobaccoVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class PenalColony(Decision):
  def registerChoices(self):
    self.addChoice('penalVerb', {
      '':100,
    })

    self.addChoice('penalNoun', {
      'penal colony':100,
      'criminal punishment':20,
    })

  def renderCode(self):
    nounPhrase = self.getChoice('penalNoun')
    nounPhrase = self.expand('OptAs') + self.expand('OptA') + nounPhrase
    self.addOrSetState('hasVerb', False)
    self.addOrSetState('verb', self.getChoice('penalVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class Explore(Decision):
  def registerChoices(self):
    self.addChoice('exploreVerb', {
      'explore':200,
      'discover':50,
      'travel':20,
      'survey':10,
      'tour':5,
      'search':5,
      'traverse':5,
      'cross':5
    })
    self.addChoice('hasUs', {
      'True': 5,
      'False':50
    })

  def renderCode(self):
    nounPhrase = ''
    if self.getChoice('hasUs') == 'True':
      nounPhrase = self.expand('Destination')
    self.addOrSetState('nounAloneOk', False)
    self.addOrSetState('verb', self.getChoice('exploreVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class SpreadReligion(Decision):
  def registerChoices(self):
    self.addChoice('spreadVerb', {
      'spread':100,
      'proselytise':5
    })

    self.addChoice('spreadNoun', {
      'religion':100,
      'their religion':100
    })

  def renderCode(self):
    nounPhrase = self.getChoice('spreadNoun')
    self.addOrSetState('nounAloneOk', False)
    self.addOrSetState('verb', self.getChoice('spreadVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class ToGetAway(Decision):
  def registerChoices(self):
    self.addChoice('getAwayVerb', {
      'get away':80,
      'leave':20
    })

  def renderCode(self):
    nounPhrase = self.expand('OptUK')
    self.addOrSetState('nounAloneOk', False)
    self.addOrSetState('verb', self.getChoice('getAwayVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class UnfairTaxation(Decision):
  def registerChoices(self):
    self.addChoice('taxVerb', {
      'escape':100,
    })

    self.addChoice('taxNoun', {
      'taxation':100,
      'taxes':100,
      'unfair taxes':20,
      'unfair taxation':20
    })

  def renderCode(self):
    nounPhrase = self.getChoice('taxNoun') + self.expand('OptUK')
    self.addOrSetState('verb', self.getChoice('taxVerb'))
    self.addOrSetState('noun', nounPhrase)
    return ''

class IncorrectPassive(Decision):
  def registerChoices(self):
    self.addChoice('passiveNoun', {
      'puritans' : 50,
      'promised cheap land' : 25,
      'colonists' : 10,
      'taxed':20,
      'criminals':20,
      'sailors':5,
      'farmers':5,
    })
  def renderCode(self):
    self.addOrSetState('verb', 'be')
    self.addOrSetState('noun', self.getChoice('passiveNoun'))
    self.addOrSetState('nounAloneOk', True)
    self.addOrSetState('mustBePast', True)
    return ''
