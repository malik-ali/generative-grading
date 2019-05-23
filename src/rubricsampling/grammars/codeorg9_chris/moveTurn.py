import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class MoveTurn(ReusableDecision):

  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'MT': 120,
      'TM': 8,
      'TMM': 2,
      'T':8,
      'M':20,
      'MTM':6,
      'MM':1,
      'MMT':1,
      'MTMT':1,
      'weeds':2
    })

  def updateRubric(self):
    style = self.getChoice(self.getDecisionName())
    if style == 'TM' or style == 'TMM':
      self.turnOnRubric('Single shape: wrong MT order')
    # note that TM doesn't count as body incorrect!
    elif style != 'MT':
      self.turnOnRubric('Single shape: body incorrect')

  def renderCode(self):
    style = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if style == 'MT': return self.renderMT()
    elif style == 'TM': return self.renderTM()
    elif style == 'TMM': return self.renderTMM()
    elif style == 'T': return self.expand('MainTurn')
    elif style == 'M': return self.expand('MainMove')
    elif style == 'MM': return self.renderMM()
    elif style == 'MTM': return self.renderMTM()
    elif style == 'MTMT': return self.renderMTMT()
    elif style == 'MMT': return self.renderMMT()
    elif style == 'weeds': return self.renderWeeds()
    raise Exception('unexpected case')

  def renderMT(self):
    code = '''
    {Move}
    {Turn}
    '''
    parts = {
      'Move':self.expand('MainMove'),
      'Turn':self.expand('MainTurn')
    }
    return gu.format(code, parts)

  def renderTM(self):
    code = '''
    {Turn}
    {Move}
    '''
    parts = {
      'Move':self.expand('MainMove'),
      'Turn':self.expand('MainTurn')
    }
    return gu.format(code, parts)

  def renderTMM(self):
    code = '''
    {Turn}
    {Move1}
    {Move2}
    '''
    parts = {
      'Move1':self.expand('MainMove'),
      'Move2':self.expand('MainMove'),
      'Turn':self.expand('MainTurn')
    }
    return gu.format(code, parts)

  def renderMM(self):
    code = '''
    {Move1}
    {Move2}
    '''
    parts = {
      'Move1':self.expand('MainMove'),
      'Move2':self.expand('MainMove')
    }
    return gu.format(code, parts)

  def renderMMT(self):
    code = '''
    {Move1}
    {Move2}
    {Turn}
    '''
    parts = {
      'Move1':self.expand('MainMove'),
      'Move2':self.expand('MainMove'),
      'Turn':self.expand('MainTurn')
    }
    return gu.format(code, parts)

  def renderMTM(self):
    code = '''
    {Move1}
    {Turn}
    {Move2}
    '''
    parts = {
      'Move1':self.expand('MainMove'),
      'Turn':self.expand('MainTurn'),
      'Move2':self.expand('MainMove')
    }
    return gu.format(code, parts)

  def renderMTMT(self):
    code = '''
    {Move1}
    {Turn1}
    {Move2}
    {Turn2}
    '''
    parts = {
      'Move1':self.expand('MainMove'),
      'Move2':self.expand('MainMove'),
      'Turn1':self.expand('MainTurn'),
      'Turn2':self.expand('MainTurn')
    }
    return gu.format(code, parts)

  def renderWeeds(self):
    return '{BodyWeeds}'

class BodyWeeds(ReusableDecision):
  def registerChoices(self):
    self.addChoice(self.getDecisionName(), {
      'A':1,
      'B':1
    })

  def updateRubric(self):
    self.turnOnRubric('Move: constant')
    self.turnOnRubric('Turn: constant')

  def renderCode(self):
    c = self.getChoice(self.getDecisionName())
    self.incrementCount()
    if c == 'A':
      return '''
      {RepeatNine}
      {RepeatSeven}
      TurnRight(130)
      MoveForward(70)
      '''
    else:
      return '''
      {RepeatNine}
      {RepeatSeven}
      {RepeatFive}
      MoveForward(30)
      TurnRight(120)
      MoveForward(30)
      TurnRight(120)
      MoveForward(30)
      '''
