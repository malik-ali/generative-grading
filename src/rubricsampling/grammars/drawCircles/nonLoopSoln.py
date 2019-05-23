import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

KEY = 'noLoopSoln-nCircles'

class NonLoopSoln(Decision):

	def registerChoices(self):
		# 10 is likely, 1 is likely, others less so
		nCirclesOptions = {
			1:50,
			10:50,
			2:2
		}
		for i in range(3, 10):
			nCirclesOptions[i] = 1
		self.addChoice(KEY, nCirclesOptions)

	def updateRubric(self):
		self.turnOnRubric('doesntUseLoop')
		nCircles = self.getChoice(KEY)
		if nCircles < 10:
			self.turnOnRubric('wrongNumCircles')

	def renderCode(self):
		nCircles = self.getChoice(KEY)
		ret = ''
		for i in range(nCircles):
			ret += self.expand('DrawCircleNoLoop', {'i':i})
		return ret