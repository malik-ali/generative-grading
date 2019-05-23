import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision

# Decision: IntType
# ------------------------
# How does the student write int
class IntType(ReusableDecision):

	def registerChoices(self):
		# may want to have an optional param id
		if not self.hasChoice(self.getKey()):
			self.addChoice(self.getKey(), {
				'int' : 100, 
				'double' : 50,
				'' : 5
			})

	def renderCode(self):
		return self.getChoice(self.getKey())

class ForIntType(ReusableDecision):

	def registerChoices(self):
		self.addChoice(self.getKey(), {
			'int': 100,
			'double':5,
			'':10
		})

	def updateRubric(self):
		if self.getChoice(self.getKey()) == '':
			self.turnOnRubric('')

	def renderCode(self):
		return self.getChoice(self.getKey())
