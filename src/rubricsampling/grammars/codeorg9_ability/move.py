import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

'''
TODO: improve the multiples for rare opps
'''
class MainMove(ReusableDecision):
  def renderCode(self):
    parts = {
      'MoveDir':self.expand('MoveDir'),
      'MoveAmt':self.expand('MoveAmt')
    }
    self.addState('firstMove', False)
    code = 'Move{MoveDir}({MoveAmt})'
    return gu.format(code, parts)

class ConstMove(ReusableDecision):
  def renderCode(self):
    # note that this won't count as using a constant
    # for a move (if a program has no moves then you
    # should turn on that flag elsewhere)
    return 'MoveForward({MoveConst})'

class MoveDir(ReusableDecision):

  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'Forward':300 * self.getState('Ability'),
      'Backwards':2
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstMove'): return
    if self.getChoice(self.getDecisionName()) == 'Backwards':
      self.turnOnRubric('Move: forward/backward confusion')

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return toReturn

class MoveAmt(ReusableDecision):
  
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'Const':120,
      'Variable':300 * self.getState('Ability')
    })


  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstMove'): return
    if self.getChoice(self.getDecisionName()) == 'Const':
      self.turnOnRubric('Move: constant')

  def renderCode(self):
    style = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if style == 'Const': return '{MoveConst}'
    return '{MoveVar}'

class MoveConst(ReusableDecision):

  def registerChoices(self):
    self.addChoice(self.getDecisionName(),{
      '100': 48, '15': 5, '30': 28, '50': 24, 
      '90': 28, '25': 6, '80': 1, '22.5': 1, 
      '60': 2, '70': 15, '51': 1, '29': 1, '10': 6, 
      '20': 2, '0': 1, '360 / 9': 1, '3': 1, '4': 1, 
      '40': 2, '5': 2, '35': 1, '36': 1, '45': 1, 
      '30 + 10': 1, '27': 3, '49': 1, '88': 2, '50 + 10': 1, 
      '75': 2, '31': 1, '68': 2, '28': 1, '1': 1, '26': 1
    })
  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return toReturn

class MoveVar(ReusableDecision):

  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '*':137 * self.getState('Ability'),
      'none':30,
      '/':12,
      '+':15,
      # 'combo':1 (can't parse this!)
    })

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstMove'): return
    opp = self.getChoice(self.getDecisionName())
    if opp == 'none':
      # note that no opp counts as wrong multiple!
      self.turnOnRubric('Move: missing opp')
      self.turnOnRubric('Move: wrong multiple')
    elif opp != '*':
      self.turnOnRubric('Move: wrong opp')

  def renderCode(self):
    oppType = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if oppType == 'none':
      return 'x'
    if oppType == 'combo':
      # this is petty rare, so i just include one example
      return '360 / x * 10'

    if oppType == '*':
      multiple = self.expand('MultCoefficient')
    if oppType == '/':
      multiple = self.expand('DivideCoefficient')
    if oppType == '+':
      multiple = self.expand('AddCoefficient')
    return self.expand('BinaryOpp', {
      'a':'x', 
      'b':multiple,
      'opp':oppType
    })

class DivideCoefficient(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '360': 5, '100': 1, '180': 1, 
      '10': 3, '5': 1
    })

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return str(toReturn)


class AddCoefficient(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      '75':3,
      '10':1
    })

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return str(toReturn)

class MultCoefficient(ReusableDecision):

  def registerChoices(self):
    options = list(range(1, 10))
    options.extend([13,20,30,110])
    optionsDict = {}
    for opt in options:
      optionsDict[opt]=1
    optionsDict[10]= 90 * self.getState('Ability')
    self.addChoice(self.getDecisionName(), optionsDict)

  def updateRubric(self):
    # only flip rubric if its the first render
    if not self.getState('firstMove'): return
    opp = self.getChoice(self.getDecisionName())
    if opp != 10:
      self.turnOnRubric('Move: wrong multiple')
    else:
      self.turnOnRubric('Move: correct')

  def renderCode(self):
    toReturn = self.getChoice(self.getDecisionName())
    self.incrementCount()
    return str(toReturn)


