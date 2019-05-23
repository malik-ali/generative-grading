import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: Increment
# ------------------------
# How does the student write var++?
# Takes a parameter, varName
class Increment(Decision):
	def registerChoices(self):
		self.addChoice('increment-style', {
			'++' : 100, 
			'+=' : 25,
			'full': 5
		})

	def renderCode(self):
		v = self.params['varName']
		c = self.getChoice('increment-style')
		templateVars = {'v':v}
		if c == '++': template = '{v}++'
		if c == '+=': template = '{v} += 1'
		if c == 'full': template = '{v} = {v} + 1'
		return gu.format(template, templateVars)