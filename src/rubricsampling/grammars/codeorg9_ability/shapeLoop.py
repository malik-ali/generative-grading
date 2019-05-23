import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

class ShapeFor(Decision):
  def registerChoices(self):
    self.addChoice('ShapeFor', {
      'standard':200 * self.getState('Ability'),
      'nestingIssue':10,
      'noLoop':33,
      'invertedRepeatFor':3,
      'loopArmsLength':25,
      'multiFor':1
    })

  def updateRubric(self):
    style = self.getChoice('ShapeFor')
    if style == 'nestingIssue':
      self.turnOnRubric('Single shape: nesting issue')
    if style == 'invertedRepeatFor':
      self.turnOnRubric('For loop: repeat instead of for')
    if style == 'loopArmsLength':
      self.turnOnRubric('For loop: armslength')

  def renderCode(self):
    style = self.getChoice('ShapeFor')
    if style == 'standard':
      return '{ShapeForStandard}'
    if style == 'noLoop':
      return '{DrawShapeX}'
    if style == 'invertedRepeatFor':
      return '{InvertedRepeatFor}'
    if style == 'loopArmsLength':
      return '{ShapeLoopArmsLength}'
    if style == 'multiFor':
      return '{MultiFor}'
    return '{NestingIssue}'


class ShapeForStandard(Decision):

  def renderCode(self):
    parts = {
      'Body':self.expand('DrawShapeX'),
      'ForHeader':self.expand('ForHeader')
    }

    code = '''{ForHeader}{{
      {Body}
    }}'''

    return gu.format(code, parts)
