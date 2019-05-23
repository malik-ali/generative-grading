import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
import random
from base import Decision, ReusableDecision


'''
To make training easier, this class is going to lightly
violate the assumption that renderCode has no randomness.
'''

class ShapeLoopArmsLengthConst(Decision):

  def updateRubric(self):
    self.turnOnRubric('Turn: constant')
    self.turnOnRubric('Move: constant')

  def renderCode(self):
    pattern = gu.mapChoice({
      '99':1,
      '97x':1, 
      '9x6':2,
      '973x4':2, 
      '975x':1,
      '974':1,
      '96':2,
      '356':1,
      '36C':1,
      '357A':1,
      '3579':1,
      'Ax':1,
      '35':1,
      '8A':1,
    })
    code = ''
    for ch in pattern:
    	code += self.expand(ch) + '\n'
    return code

  def expand(self, ch):
  	if ch == 'x': return '{ConstMoveTurn}'
  	if ch == 'A': return self.expandRepeat(10)
  	if ch == 'C': return self.expandRepeat(12)
  	return self.expandRepeat(int(ch))

  def expandRepeat(self, x):
  	code = '''
  	Repeat({x}){{
  		{body}
  	}}
  	'''
  	params = {
  		'x':x,
  		'body':'{Repeat'+ self.toString(x) + '}'
  	}
  	v = gu.format(code, params)
  	return v

  def toString(self, v):
  	m = {
  		3:'Three',4:'Four',5:'Five',6:'Six',
  		7:'Seven',8:'Eight',9:'Nine',10:'Ten',
  		12:'Twelve'
  	}
  	return m[v]

class ConstMoveTurn(Decision):
	def renderCode(self):
		pattern = gu.mapChoice({
			'M':1,
			'T':1,
			'MT':10,
			'MTM':1
		})

		toReturn = ''
		for ch in pattern:
			if ch == 'M':
				toReturn += self.constMove() + '\n'
			if ch == 'T':
				toReturn += self.constTurn() + '\n'
		return toReturn

	def constMove(self):
		v = str(random.randint(1, 360))
		return 'MoveForward(' + v + ')'

	def constTurn(self):
		v = str(random.randint(1, 360))
		return 'TurnRight(' + v + ')'

class RepeatTwelve(Decision):
  def renderCode(self):
    return 'TurnLeft(70)\nMoveForward(100)'

class RepeatTen(Decision):
  def renderCode(self):
    return gu.mapChoice({
      'MoveForward(90)\nTurnRight(40)':2,
      'MoveForward(100)\nTurnRight(36)':1
    })

class RepeatNine(Decision):
  def renderCode(self):
    return gu.mapChoice({
      'MoveForward(90)\nTurnRight(40)':9,
      'MoveForward(92)\nTurnRight(41)':1,
      'MoveForward(100)\nTurnRight(90)':1,
      'MoveForward(36)\nTurnRight(80)':1,
      '''MoveForward(90)
         TurnRight(40)
         MoveForward(90)
         TurnRight(40)''':1
    })

class RepeatEight(Decision):
  def renderCode(self):
    return 'TurnLeft(70)\nMoveForward(80)'

class RepeatSeven(Decision):
  def renderCode(self):
    return gu.mapChoice({
      'MoveForward(68)\nTurnRight(51)':1,
      'MoveForward(100)\nTurnRight(90)':1,
      'MoveForward(70)\nTurnRight(51.5)':2,
      'MoveForward(100)\nTurnRight(51.4)':1,
      'MoveForward(100)\nTurnRight(51.4)':1,
      'MoveForward(70)\nTurnRight(51)':1,
      'MoveForward(130)\nTurnRight(70)':1
    })

class RepeatSix(Decision):
	def renderCode(self):
		return gu.mapChoice({
      'MoveForward(68)\nTurnRight(55)':1,
      'MoveForward(70)\nTurnRight(51.32)':1,
      'MoveForward(50)\nTurnRight(72)':1,
      'MoveForward(70)\nTurnRight(42)':1,
      'MoveForward(100)\nTurnRight(80)':1,
      'MoveForward(70)\nTurnRight(51)':1
    })

class RepeatFive(Decision):
	def renderCode(self):
		return gu.mapChoice({
      'MoveForward(50)\nTurnRight(72)':3,
      'MoveForward(50)\nTurnRight(90)':1,
      'MoveForward(100)\nTurnRight(90)':1
    })
class RepeatFour(Decision):
	def renderCode(self):
		return gu.mapChoice({
      'MoveForward(50)\nTurnRight(75)':1,
      'MoveForward(50)\nTurnRight(160)':1,
      'MoveForward(72)\nTurnRight(50)':1
    })

class RepeatThree(Decision):
	def renderCode(self):
		return gu.mapChoice({
      'MoveForward(30)\nTurnRight(120)':7,
      'MoveForward(100)\nTurnRight(90)':1,
    })