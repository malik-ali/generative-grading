import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Soln
# ------------------------
# I think that everyone will use the count up solution..
class ForSoln(Decision):
	def registerChoices(self):
		self.addChoice('forloop-style', {
			'countI' : 100,
			# 'countX' : 20
		})

	def renderCode(self):
		return self.expand(self.getLoopBody())

	def getLoopBody(self):
		style = self.getChoice('forloop-style')
		if style == 'countI': return 'ForCountI'
		if style == 'countX': return 'ForCountX'
		