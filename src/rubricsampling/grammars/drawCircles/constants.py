import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Constants(Decision):
	
	def renderCode(self):
		constMap = self.getState('constants')
		return self.expand('PermuteDict', {'dict': constMap, 'id':'const'})
