import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Constants(Decision):

	def registerChoices(self):
		self.addChoice('constant-starterCode', {
			True : 100, 
			False : 5
		})

	def renderCode(self):
		if self.getChoice('constant-starterCode'):
			return 'private static final int START = 10;'

		hasConst = self.hasChoice('usesConstant') and self.getChoice('usesConstant')
		if hasConst:
			name = self.getChoice('constName')
			return gu.format('private static final int {n} = 10;', {'n':name})
		return ''

class ConstName(Decision):
	def registerChoices(self):
		if self.hasChoice('constName'):
			return
		self.addChoice('constName', {
			'START' : 100
		})

	def renderCode(self):
		return self.getChoice('constName')