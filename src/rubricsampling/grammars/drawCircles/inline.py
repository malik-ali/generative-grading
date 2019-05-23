import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision

'''
Expected params:
id: a unique identifier for the inline
exp: the expression that may be inlined
stateKey: the location where the temp variable will be declared 
  OR a list of locations.
var: the variable name, if not inlined
'''
class Inline(ReusableDecision):

	def registerChoices(self):
		self.addChoice(self.getKey(), {
			True:100,
			False:100
		})	


	def renderCode(self):
		exp = self.params['exp']
		loc = self.params['stateKey']
		var = self.params['var']
		isInline = self.getChoice(self.getKey())

		if isInline:
			return exp
		else:
			# also make sure its predeclared
			varType = self.params['varType']
			declareLine = varType + ' ' + var + ' = ' + exp + ';'
			location = self.getState(loc)
			if type(location) == type({}):
				location[var] = declareLine 
			elif type(location) == type([]):
				location.append(declareLine)
			else:
				raise Exception('unknown loc type')
			# but your are just going to render the var name
			return var


		


