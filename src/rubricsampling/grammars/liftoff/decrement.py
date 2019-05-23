import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: Decrement
# ------------------------
# How does the student write var--?
# Takes a parameter, varName
class Decrement(Decision):
	def registerChoices(self):
		self.addChoice('decrement-style', {
			'--' : 100, 
			'-=' : 50,
			'full': 25
		})

	def renderCode(self):
		v = self.params['varName']
		c = self.getChoice('decrement-style')
		templateVars = {'v':v}
		if c == '--': template = '{v}--'
		if c == '-=': template = '{v} -= 1'
		if c == 'full': template = '{v} = {v} - 1'
		return gu.format(template, templateVars)