import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class MaxToken(Decision):

	def registerChoices(self):
		if not self.hasChoice('usesConstant'):
			self.addChoice('usesConstant', {
				True:100,
				False:50
			})

	def renderCode(self):
		if self.getChoice('usesConstant'):
			return self.expand('ConstName')
		else:
			return '10'