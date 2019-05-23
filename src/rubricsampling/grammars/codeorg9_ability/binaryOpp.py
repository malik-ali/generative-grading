import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision


class BinaryOpp(ReusableDecision):

	def registerChoices(self):
		if self.hasChoice(self.getDecisionName()):
			return
			
		self.addChoice(self.getDecisionName(), {
			'forward':100,
			'backward':20
		})

	def renderCode(self):
		direction = self.getChoice(self.getDecisionName())

		a = str(self.params['a'])
		b = str(self.params['b'])
		opp = str(self.params['opp'])

		# you need to increment the counter so that its reusable
		self.setState('BinaryOpp_count', self.getCount() + 1)

		if direction == 'forward':
			return a + ' ' + opp + ' ' + b
		else:
			return b + ' ' + opp + ' ' + a


		


