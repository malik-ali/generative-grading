import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


class ShapeLoopArmsLength(Decision):
  def getChoiceName(self):
    return 'ShapeLoopArmsLength'

  def registerChoices(self):
    self.addChoice(self.getChoiceName(), {
      'preFor':4,
      'postFor':2,
      'const':10,
      'weeds':2
    })

  def renderCode(self):
    style = self.getChoice(self.getChoiceName())
    if style == 'preFor':
      return '{ShapeLoopArmsLengthPre}'
    if style == 'postFor':
      return '{ShapeLoopArmsLengthPost}'
    if style == 'const':
      return '{ShapeLoopArmsLengthConst}'
    if style == 'weeds':
      return '{ShapeLoopArmsLengthWeeds}'

class ShapeLoopArmsLengthPre(Decision):
  def renderCode(self):
    return '''
      {ConstMove}
      {ShapeForStandard}
    '''

class ShapeLoopArmsLengthWeeds(Decision):
  def getChoiceName(self):
    return 'ShapeLoopArmsLengthWeeds'

  def updateRubric(self):
    self.turnOnRubric('Move: constant')
    self.turnOnRubric('Turn: constant')

  def renderCode(self):
    style = self.getChoice(self.getChoiceName())
    if style == 'A':
      return '''{ForHeaderPixels1}{{
        MoveForward(90)
        TurnRight(40)
      }}
      TurnLeft(220)
      MoveForward(90)
      {ForHeaderPixels}{{
        TurnLeft(65)
        MoveForward(80)
      }}'''
    if style == 'B':
      return '''Repeat(3){{
        MoveForward(30)
        TurnRight(120)
      }}
      {ForHeaderPixels}{{
        MoveForward(50)
        TurnRight(71.5)
      }}
      '''

  def registerChoices(self):
    # this is rare so i include three examples
    self.addChoice(self.getChoiceName(), {
      'A':1,
      'B':1
    })

class ShapeLoopArmsLengthPost(Decision):
  def getChoiceName(self):
    return 'ShapeLoopArmsLengthPre'

  def renderCode(self):
    style = self.getChoice(self.getChoiceName())
    if style == 'A':
      return '''
        {ShapeForStandard}
        {RepeatFive}
        {RepeatSeven}
        TurnRight(2)
        {RepeatNine}
        '''
    if style == 'B':
      return '''
        {ShapeForStandard}
        For(5, 2, 10){{
          Repeat(x){{
            {MoveTurn}
          }}
        }}
        MoveForward(72)
        TurnRight(65)
        MoveForward(100)
      '''
    if style == 'C':
      return '''
        {ShapeForStandard}
        For(3, 9, 1){{
          {MainTurn}
        }}
        '''

  def registerChoices(self):
    # this is rare so i include three examples
    self.addChoice(self.getChoiceName(), {
      'A':1,
      'B':1,
      'C':1
    })
    