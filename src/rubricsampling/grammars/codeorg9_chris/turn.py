import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class MainTurn(ReusableDecision):
  def renderCode(self):
    parts = {
      'TurnDir':self.expand('TurnDir'),
      'TurnAmt':self.expand('TurnAmt')
    }
    self.addState('firstTurn', False)
    code = 'Turn{TurnDir}({TurnAmt})'
    return gu.format(code, parts)

class TurnDir(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'Right':250,
      'Left':15
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstTurn'): return
    isRight = self.getChoice(self.getDecisionName()) == 'Right'
    if not isRight:
      self.turnOnRubric('Turn: left/right confusion')

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return toReturn

class TurnAmt(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'Const': 120,
      'Variable': 300
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstTurn'): return
    isConst = self.getChoice(self.getDecisionName()) == 'Const'
    if isConst:
      self.turnOnRubric('Turn: constant')

  def renderCode(self):
    style = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if style == 'Const': return '{TurnConst}'
    return '{TurnVar}'

class TurnConst(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(),{
      '39': 2, '40': 31, '51.42': 1, '72': 9, '120': 22, 
      '90': 32, '130': 3, '60': 9, '360 / 10': 4, '55': 2, 
      '89': 3, '220': 1, '65': 1, '3': 1, '70': 3, '75': 4, 
      '140': 1, '36': 4, '71.5': 1, '360 / 5': 2, '9': 1, 
      '51.5': 2, '45': 3, '80': 3, '240': 3, '52': 1, 
      '30': 3, '51': 3, '150': 1, '42': 1, '360': 1, '51.4': 1, 
      '50': 3, '51.35': 1, '160': 1, '51.32': 1, '4': 1, 
      '360 / 29': 1, '25': 1, '360 / 2': 1, '180': 1, '360 / 22': 1, 
      '10': 1, '79': 1, '2': 1, '41': 1, '20': 1, '360 / 8': 1, '360 / 4': 1
    })
  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return toReturn

class TurnVar(ReusableDecision):

  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '*':137,
      'none':30,
      'k/x':12,
      'x/k':12,
      '+':2,
      '-':5
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstTurn'): return
    opp = self.getChoice(self.getDecisionName())
    if opp == 'none':
      # note that no opp does not counts as wrong multiple!
      self.turnOnRubric('Turn: missing opp')
    elif opp != 'k/x':
      self.turnOnRubric('Turn: wrong opp')

  def renderCode(self):
    oppType = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if oppType == 'none':
      return 'x'
    if oppType == 'k/x':
      return self.expand('TurnXKCoefficient') + ' / x'
    if oppType == 'x/k':
      return 'x / 360'

    if oppType == '*':
      multiple = '10'
    if oppType == '-':
      multiple = '10'
    if oppType == '+':
      multiple = '7'
    return self.expand('BinaryOpp', {
      'a':'x', 
      'b':multiple,
      'opp':oppType
    })

class TurnXKCoefficient(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '360':100,
      '10': 2, '180': 3, '279': 1, 
      '365': 2, '355': 1, '350': 1
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstTurn'): return
    v = self.getChoice(self.getDecisionName())
    if v != '360':
      self.turnOnRubric('Turn: wrong multiple')

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return toReturn


