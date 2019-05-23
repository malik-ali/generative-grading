import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: PrintVar
# ------------------------
# How does the student write println(varName)?
# Takes a parameter, varName
class PrintVar(Decision):
	def registerChoices(self):
		self.addChoice('println-function', {
			'print' : 1, 
			'println' : 100,
			'none': 1,
		})

	def renderCode(self):
		paramStr = self.expand('PrintVarParam', self.params)
		c = self.getChoice('println-function')
		templateVars = {'p':paramStr}
		if c == 'print': template = 'print({p});'
		if c == 'println': template = 'println({p});'
		if c == 'none': template = ''
		ret = gu.format(template, templateVars)
		return ret

class PrintVarParam(Decision):
	def renderCode(self):
		return self.params['varName']
	# def registerChoices(self):
	# 	self.addChoice('rubric-println-noQuotes', {
	# 		True : 100, 
	# 		False : 5
	# 	})

	# def renderCode(self):
	# 	v = self.params['varName']
	# 	c = self.getChoice('rubric-println-noQuotes')
	# 	if c:
	# 		return v
	# 	else:
	# 		return '"' + v + '"'
