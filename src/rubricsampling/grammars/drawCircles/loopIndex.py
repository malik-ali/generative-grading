import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: LoopIndex
# ------------------------
# How does the student write i for a loop
class LoopIndex(Decision):
	def registerChoices(self):
		# not everyone uses i, but for now we are
		# anonimizing variable names
		self.addChoice('loop-indexName', {
			'i' : 100, 
			# 'j' : 10,
			# 'c': 5,
			# 'v': 5,
			# 'n': 5,
			# 'counter': 1
		})

	def renderCode(self):
		name = self.getChoice('loop-indexName')
		return name