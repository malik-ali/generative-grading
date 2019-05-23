import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision

# anytime a user uses a magic number, there is a change
# for an off by one.
# params: key, target
class Value(ReusableDecision):

	def registerChoices(self):
		# are they off by one?
		self.addChoice(self.getKey(), {
			'large': 5, 
			'small': 5,   
			'false' : 100
		})

	def renderCode(self):
		target = self.params['target']
		offByOne = self.getChoice(self.getKey())

		val = target
		if offByOne == 'large': val += 1
		if offByOne == 'small': val -= 1

		return str(val)