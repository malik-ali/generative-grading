import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Program(Decision):
	def registerChoices(self):
		'''
		We add a dummy RV with only one value 
		to make the RNN code easier.
		'''				
		self.addChoice(self.ROOT_RV_NAME, {
			self.ROOT_RV_VAL: 1
		})

	def renderCode(self):
		# a few data lists that other expands could add to
		self.addState('methodList', [])
		self.addState('constants', {})
		self.addState('instanceVars', [])

		# this is an example of rendering out of order
		# you need to expand methods first (and in particular)
		# the run method. Then you can expand ivars etc
		templateVars = {
			'Solution':self.expand('Solution'),
			'HelperMethods':self.expand('HelperMethods'),
			'Constants':self.expand('Constants')
		}
		template = '''
			{Constants}
			public void run() {{
				{Solution}
			}}
			{HelperMethods}
		'''
		return gu.format(template, templateVars)