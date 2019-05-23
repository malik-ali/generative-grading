
import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Count Down
# ------------------------
# This is for people who have a for loop with -- as the
# for loop operator.
'''
for(int var0 = 11; var0 > 1; var0--) {
  println(var0 - 1);
}
'''
class ForCountDown(Decision):
	def registerChoices(self):
		# does the student use >= or > in the for loop
		self.addChoice('countDown-compOpp', {
			'>=' : 50, 
			'>' : 50,
			'<' : 5,
			'<=' : 5
		})
		# are they off by one on last value
		self.addChoice('rubric-loop-endOffByOne', {
			'endPrint2': 10, # the last number printed is a 2
			'endPrint0': 70,  # the last number printed is a 0 
			False : 100
		})

	def updateRubric(self):
		opp = self.getChoice('countDown-compOpp')
		if not opp.startswith('>'):
			self.turnOnRubric('incorrectCompOpp')

	def preregisterDecisionIds(self):
		return {
			'ForIntType': {'loopStartType'}
		}

	def renderCode(self):
		v = self.expand('LoopIndex')
		templateVars = {
			'v': v,
			'comp': self.getChoice('countDown-compOpp'),
			'endVal': self.getEndVal(),
			'm': self.expand('MaxToken'),
			'dec': self.expand('Decrement', {'varName':v}),
			'printVar':self.expand('PrintVar', {'varName':v}),
			'varType':self.expand('ForIntType', {'id': 'loopStartType'})
		}
		template = """
		for({varType} {v} = {m}; {v} {comp} {endVal}; {dec}) {{
			{printVar}
		}}
		"""
		return gu.format(template, templateVars)


	def getEndVal(self):
		opp = self.getChoice('countDown-compOpp')
		offByOne = self.getChoice('rubric-loop-endOffByOne')

		val = 1
		if offByOne == 'endPrint0': val = 0
		if offByOne == 'endPrint2': val = 2

		if opp == '>': val -= 1
		return val