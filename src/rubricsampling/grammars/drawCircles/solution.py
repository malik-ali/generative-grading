import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Solution(Decision):
	def registerChoices(self):
		self.addChoice('hasSoln', {
			True:100,
			False:2
		})

	def updateRubric(self):
		if not self.getChoice('hasSoln'):
			self.turnOnRubric('noSolution')

	def renderCode(self):
		# we might want to consider allowing this to be decomposed...
		return '{DrawCircles}'