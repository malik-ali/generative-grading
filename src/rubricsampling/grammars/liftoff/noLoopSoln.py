import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class NoLoopSoln(Decision):

	def registerChoices(self):
		# 10 is likely, 1 is likely, others less so
		nLinesOptions = {
			10: 100,
			1: 50,
			2: 2
		}
		
		for i in range(3, 10):
			nLinesOptions[i] = 1

		self.addChoice('noLoopSoln-nLines', nLinesOptions)

	def renderCode(self):
		nLines = self.getChoice('noLoopSoln-nLines')
		ret = ''
		curr = 10
		for i in range(nLines):
			ret += 'println("'+str(curr)+'");\n'
			curr -= 1
		return ret