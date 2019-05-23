import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


'''
People with armslength shapes often use constants...
Which could count as a rubric item. And should likely
change their probability of using constants in the body

Similarly, people with armsUnrolled hardly ever put it in 
a for loop...
'''

class DrawShapeX(Decision):
  def registerChoices(self):
    self.addChoice('drawShape-style', {
      'standard':150 * self.getState('Ability'),
      'armsAfter':13,
      'armsBefore':3,
      'armsUnrolled':2,
    })

  def updateRubric(self):
    if self.getChoice('drawShape-style') != 'standard':
      self.turnOnRubric('Single shape: armslength')

  def renderCode(self):
    style = self.getChoice('drawShape-style')
    if style == 'armsUnrolled': return '{DrawShapeUnrolled}'

    core = self.expand('DrawShapeStandard')
    if style == 'armsBefore':
      before = self.expand('DrawShapeArmsBefore')
      return before + '\n' + core

    if style == 'armsAfter':
      after = self.expand('DrawShapeArmsAfter')
      return core + '\n' + after

    if style == 'standard':
      return core
    
    raise Exception('unexpected case')

class DrawShapeStandard(Decision):
  def registerChoices(self):
    self.addChoice('DrawShapeStandard', {
      'hasRepeat':200 * self.getState('Ability'),
      'noRepeat':20
    })

  def updateRubric(self):
    # note: a missing repeat will be found in "Program"
    pass

  def renderCode(self):
    style = self.getChoice('DrawShapeStandard')
    if style == 'hasRepeat':
      parts = {
        'Body':self.expand('MoveTurn'),
        'Iter':self.expand('DrawShapeXIter')
      }
      code = '''Repeat({Iter}){{
        {Body}
      }}'''
      return gu.format(code, parts)
    else:
      return '{MoveTurn}'

class DrawShapeXIter(Decision):
  def registerChoices(self):
    self.addChoice('DrawShapeX-iter', {
      'x': 100 * self.getState('Ability'),
      '9': 4, 'x * 10': 1, '4': 5, '3': 4, '1': 2, 
      'x + 1': 1, 'x / 2': 1, '10': 1, 
      'x * 0': 1, '360 / x': 1
    })

  def updateRubric(self):
    if self.getChoice('DrawShapeX-iter') != 'x':
      self.turnOnRubric('Single shape: wrong iter #')

  def renderCode(self):
    return self.getChoice('DrawShapeX-iter')

class DrawShapeArmsAfter(Decision):
  def renderCode(self):
    return self.getChoice('DrawShapeArmsAfter')

  def registerChoices(self):
    self.addChoice('DrawShapeArmsAfter', {
      'MoveForward(x * 10)':1,
      'MoveForward(100)':2,
      '''Repeat(x){{
        MoveForward(60)
      }}''':1,
      '''MoveForward(50)
      TurnRight(90)''':1,
      '''
      TurnLeft(40)
      MoveBackwards(20)
      TurnRight(30)
      MoveForward(70)
      ''':1,
      'TurnRight(120)':1,
      '''TurnRight(75)
      MoveForward(100)''':1,
      '''MoveForward(75)
      TurnRight(60)''':1,
      '''MoveForward(70)
      TurnRight(50)
      Repeat(6){{
        MoveForward(70)
        TurnRight(51.35)
      }}''':1,
      '''MoveForward(70)
      TurnRight(50)
      Repeat(6){{
        MoveForward(70)
        TurnRight(51.32)
      }}''':1,
      'TurnRight(x)':1,
      '''
      Repeat(x){{
        MoveForward(x + 10)
      }}''':1
    })

class DrawShapeArmsBefore(Decision):
  def renderCode(self):
    return self.getChoice('DrawShapeArmsBefore')

  def registerChoices(self):
    self.addChoice('DrawShapeArmsBefore', {
      'MoveForward(x * 10)':1,
      'MoveForward(30)':1
    })

class DrawShapeUnrolled(Decision):
  def renderCode(self):
    return self.getChoice('DrawShapeUnrolled')

  def registerChoices(self):
    self.addChoice('DrawShapeUnrolled', {
      '''
      MoveForward(50)
      TurnRight(70)
      MoveForward(50)
      TurnRight(75)
      MoveForward(50)
      TurnRight(72)
      MoveForward(50)
      TurnRight(72)
      MoveForward(50)
      TurnRight(140)
      MoveForward(25)
      ''':1,
      '''
      TurnRight(60)
      MoveForward(30)
      TurnLeft(120)
      MoveForward(30)
      ''':1,
      '''
      MoveForward(90)
      TurnRight(40)
      MoveForward(90)
      TurnRight(40)
      MoveForward(100)
      ''':1,
      })
