import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Soln
# ------------------------
# This is for people who solved the problem using a
# for loop.
class ForSoln(Decision):
	def registerChoices(self):
		self.addChoice('forloop-style', {
			'countUp' : 30, 
			'countDown' : 70
		})

	def renderCode(self):
		style = self.getChoice('forloop-style')
		if style == 'countUp': return '{ForCountUp}'
		if style == 'countDown': return '{ForCountDown}'