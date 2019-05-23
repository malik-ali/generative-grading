import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

class ForHeader(Decision):
  def getChoiceName(self):
    return 'ForHeader'

  def registerChoices(self):
    self.addChoice(self.getChoiceName(), {
      'byPixels':40,
      'bySides':150
    })

  def updateRubric(self):
    style = self.getChoice(self.getChoiceName())
    if style == 'byPixels':
      self.turnOnRubric('For loop: not looping by sides')

  def renderCode(self):
    style = self.getChoice(self.getChoiceName())
    if style == 'byPixels':
      return '{ForHeaderPixels}'
    else:
      return '{ForHeaderSides}'

class ForHeaderSides(Decision):
  def renderCode(self):
    return 'For({ForSidesStart}, {ForSidesEnd}, {ForSidesDelta})'

#################################

class ForSidesStart(Decision):
  def registerChoices(self):
    self.addChoice('ForSidesStart', {
      '3': 200,
      '1': 44, '2': 11, '10': 7, '0': 4, 'x': 1, '5': 13, 
      '9': 6, '4': 4
    })

  def updateRubric(self):
    if self.getChoice('ForSidesStart') != '3':
      self.turnOnRubric('For loop: wrong start')

  def renderCode(self):
    return self.getChoice('ForSidesStart')

# either 9 or 10 are fine
class ForSidesEnd(Decision):
  def registerChoices(self):
    self.addChoice('ForSidesEnd', {
      '9':100, '10':100,
      '1': 6, '5': 7, '4': 13, '10': 3, '2': 6, 
      '20': 3, '8': 9, 'x': 1, '7': 4, '6': 5, 
      '84': 1, '3': 5, '15': 3, '100': 6, '11': 2, 
      '25': 1, '0': 1, '12': 2, '60': 1
    })

  def updateRubric(self):
    end = self.getChoice('ForSidesEnd')
    if end != '9' and end != '10':
      self.turnOnRubric('For loop: wrong end')

  def renderCode(self):
    return self.getChoice('ForSidesEnd')

# only 2 is ok
class ForSidesDelta(Decision):
  def registerChoices(self):
    self.addChoice('ForSidesDelta', {
      '2': 200,
      '1': 46, '10': 38, '5': 6, 
      '3': 28, 'x': 1, '43': 1, '4': 3, '1.5': 1, 
      '9': 4, '14': 1, '40': 1, '7': 1, '0.5': 1
    })

  def updateRubric(self):
    delta = self.getChoice('ForSidesDelta')
    if delta != '2':
      self.turnOnRubric('For loop: wrong delta')

  def renderCode(self):
    return self.getChoice('ForSidesDelta')

###################################

class ForHeaderPixels(ReusableDecision):

  def renderCode(self):
    choice = self.getChoice(self.getDecisionName())
    choiceParts = choice.split(',')
    params = {
      'Start':choiceParts[0],
      'End':choiceParts[1],
      'Delta':choiceParts[2]
    }
    code = 'For({Start}, {End}, {Delta})'
    self.incrementCount()
    return gu.format(code, params)

  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '1,100,10': 7, 
      '100,25,25': 1, 
      '50,100,10': 2, 
      '10,100,10': 4, 
      '25,100,10': 3, 
      '100,200,100': 1, 
      '20,100,20': 1, 
      '30,90,20': 2, 
      '3,100,10': 1, 
      '25,100,20': 1, 
      '1,90,10': 1, 
      '9*10,3*10,1': 1, 
      '1,360,60': 1, 
      '10,36,10': 1, 
      '90,100,10': 1, 
      '30,240,20': 1, 
      '30,40,10': 1, 
      '10,400,23': 1, 
      '30,120,30': 1, 
      '25,50,25': 1, 
      '1,25,10': 1, 
      '10,100,1000': 1, 
      '15,150,15': 1
    })

