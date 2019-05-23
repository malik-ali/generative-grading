import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

class ChoosePriors(Decision):
  def registerChoices(self):
    self.addChoice('Ability', {
      10.0:1,
      5.0:4,
      2.0:5,
      1.0:6,
      0.5:2
    })

  def renderCode(self):
    '''
    Ability is a variable which is sotred in state, not rendered
    '''
    self.addState('Ability', self.getChoice('Ability'))
    return ''
