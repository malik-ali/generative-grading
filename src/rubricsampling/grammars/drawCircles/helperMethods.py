import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: HelperMethods
# ------------------------
# Render all the helper methods that have been added to the engine
# TODO: we should make the permutation a choice (where the key is
# an index vector). Right now "random" breaks the promise of render
class HelperMethods(Decision):
	# I would like to come up with a better way of
	# articulating the ordering invariance
	def registerChoices(self):
		self.addChoice('method-order', {
			'forward' : 100, 
			'reverse' : 10,
			'random'  : 10, # WARNING: breaks promise of consistency
		})

	def renderCode(self):
		ret = ''
		allMethods = self.getState('methodList')

		order = self.getChoice('method-order')
		if order == 'reverse': allMethods.reverse()
		if order == 'random': random.shuffle(allMethods) # THIS IS RANDOM

		for method in allMethods:
			ret += self.makeMethod(method)
		return ret

	def makeMethod(self, method):
		# method is a tuple (name, body)
		name, body = method
		templateVars = {
			'name':name,
			'body':body
		}
		template = '''
		private void {name}() {{
			{body}
		}}
		'''
		return gu.format(template, templateVars)

