import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import numpy as np
from base import Decision, ReusableDecision


'''
This class captures that students can be on an early stage
of development where they don't yet have a final answer
'''
class Answer(Decision):

	def registerChoices(self):
		self.addChoice('Stage', {
			'oneConstMove':10,
			'oneConstTurn':1,
			'oneConstBody':10,
			'oneConstShape':5,
			'shapeLoop':300,
		})

	def renderCode(self):
		stage = self.getChoice('Stage')
		# most common is to be on the final stage
		if stage == 'shapeLoop':
			return '{ShapeFor}'

		if stage == 'oneConstMove':
			return '{ConstMove}'
		if stage == 'oneConstTurn':
			return '{ConstTurn}'

		if stage == 'oneConstBody':
			return '{OneConstBody}'
		if stage == 'oneConstShape':
			return '{OneConstShape}'
		raise Exception('unknown stage')

	def updateRubric(self):
		stage = self.getChoice('Stage')
		if stage == 'oneConstMove':
			self.turnOnRubric('Move: constant')
		if stage == 'oneConstTurn':
			self.turnOnRubric('Turn: constant')

class OneConstShape(Decision):
	def registerChoices(self):
		self.addChoice('OneConstShape', {
			'3':10,
			'9':2
		})

	def renderCode(self):
		n = self.getChoice('OneConstShape')
		return '''
		Repeat('''+n+''') {{
			{OneConstBody}
		}}
		'''

class OneConstBody(Decision):
	def registerChoices(self):
		self.addChoice('OneConstBody', {
			'MT':10 * self.getState('Ability'),
			'MTM':1,
			'TM':1
		})

	def updateRubric(self):
		self.turnOnRubric('Move: constant')
		self.turnOnRubric('Turn: constant')
		if self.getChoice('OneConstBody') == 'MTM':
			self.turnOnRubric('Single shape: body incorrect')

	def renderCode(self):
		pattern = self.getChoice('OneConstBody')

		toReturn = ''
		for ch in pattern:
			if ch == 'M':
				toReturn += '{ConstMove}\n'
			if ch == 'T':
				toReturn += '{ConstTurn}\n'

		return toReturn[:-1]
