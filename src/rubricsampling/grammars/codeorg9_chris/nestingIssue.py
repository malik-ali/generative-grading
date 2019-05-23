import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision

class NestingIssue(Decision):
  def registerChoices(self):
    self.addChoice('NestingIssue', {
      'FMRT':4,
      'FRMT':3,
      'FRMT2':1,
      'RMFTM':1
    })

  def renderCode(self):
    style = self.getChoice('NestingIssue')
    if style == 'FMRT': return self.renderFMRT()
    elif style == 'FRMT': return self.renderFRMT()
    elif style == 'FRMT2': return self.renderFRMT2()
    elif style == 'RMFTM':return self.renderRMFTM()

  def getParts(self):
    return {
      'ForHeader':self.expand('ForHeader'),
      'Move':self.expand('MainMove'),
      'Turn':self.expand('MainTurn'),
      'Iter':self.expand('DrawShapeXIter')
    }
  
  def renderFMRT(self):
    parts = self.getParts()
    code = '''{ForHeader}{{
      {Move}
      Repeat({Iter}) {{
        {Turn}
      }}
    }}'''
    return gu.format(code, parts)

  def renderFRMT(self):
    parts = self.getParts()
    code = '''{ForHeader}{{
      Repeat({Iter}) {{
        {Move}
      }}
      {Turn}
    }}'''
    return gu.format(code, parts)

  def renderFRMT2(self):
    parts = self.getParts()
    code = '''{ForHeader}{{
      Repeat({Iter}) {{
        {Move}
      }}
    }}
    {Turn}'''
    return gu.format(code, parts)

  def renderRMFTM(self):
    parts = {
      'ForHeader':self.expand('ForHeader'),
      'Move1':self.expand('MainMove'),
      'Move2':self.expand('MainMove'),
      'Turn':self.expand('MainTurn'),
      'Iter':self.expand('DrawShapeXIter')
    }
    code = '''
    Repeat({Iter}) {{
      {Move1}
    }}
    {ForHeader}{{
      {Turn}
    }}
    {Move2}
    '''
    return gu.format(code, parts)

class InvertedRepeatFor(Decision):
  def updateRubric(self):
    # this counds as a misuse of Repeat and can
    # still trigger a "missing repeat" tag
    self.turnOnRubric('Single shape: missing repeat')

  def renderCode(self):
    parts = {
      'ForHeader':self.expand('ForHeader'),
      'MoveTurn':self.expand('MoveTurn'),
      'Iter':self.expand('DrawShapeXIter')
    }
    code = '''
    Repeat({Iter}) {{
      {ForHeader}{{
        {MoveTurn}
      }}
    }}
    '''
    return gu.format(code, parts)

class MultiFor(Decision):
  def updateRubric(self):
    self.turnOnRubric('Move: constant')
    self.turnOnRubric('For loop: not looping by sides')
    self.turnOnRubric('Single shape: missing repeat')
  def renderCode(self):
    return '''
For(1, 100, 10){{
  For(1, 100, 10){{
    For(1, 100, 10){{
      For(1, 100, 10){{
        MoveForward(100)
      }}
    }}
  }}
}}
    '''
