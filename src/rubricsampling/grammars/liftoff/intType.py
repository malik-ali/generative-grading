import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: IntType
# ------------------------
# How does the student write int
class IntType(Decision):
	def registerChoices(self):
		self.addChoice('correctVarType', {
			True : 100, 
			False : 10
		})

	def renderCode(self):
		c = self.getChoice('correctVarType')
		if c: return 'int'
		else: return 'double'