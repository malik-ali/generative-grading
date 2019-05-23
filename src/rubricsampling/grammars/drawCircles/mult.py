import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision

'''
WARNING: 
How do we allow for this decision to be made
differenently in different contexts? How does
this impact our ability to rationalize about
'''
class Mult(ReusableDecision):

	def registerChoices(self):
		if self.hasChoice(self.getKey()):
			return
			
		self.addChoice(self.getKey(), {
			'forward':100,
			'backward':50
		})

	def renderCode(self):
		direction = self.getChoice(self.getKey())

		a = str(self.params['a'])
		b = str(self.params['b'])

		if direction == 'forward':
			return a + '*' + b
		else:
			return b + '*' + a
		


