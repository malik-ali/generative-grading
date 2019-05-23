
import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Y(Decision):
	def registerChoices(self):
		self.addChoice('yLoc', {
			'loop-start-lines':100,
			'loop-pre-vars':100
		})

	def preregisterDecisionIds(self):
		return {
			'Inline': {'inline-y'},
			'IntType': {'yType'}
		}

	def renderCode(self):
		exp = self.expand('ZeroY')
		varType = self.expand('IntType', {'id':'yType'})
		loc = self.getChoice('yLoc')
		if not self.hasState(loc):
			return exp

		return self.expand('Inline', {
				'id': 'inline-y', 
				'exp': exp, 
				'stateKey': loc,
				'var': 'y',
				'varType': varType
		})