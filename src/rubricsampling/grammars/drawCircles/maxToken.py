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

	def preregisterDecisionIds(self):
		return {
			'IntType': {'maxConstType'}
		}			

	def renderCode(self):
		if self.getChoice('usesConstant'):
			varType = self.expand('IntType', {'id':'maxConstType'})
			line = 'private static final {} N_CIRCLES = 10;'.format(varType)
			self.getState('constants')['N_CIRCLES'] = line
			return 'N_CIRCLES'
		else:
			return '10'