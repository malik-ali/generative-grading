
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

	def preregisterDecisionIds(self):
		return {
			'Value': {'rubric-loop-end-offByOne'}
		}		

	def updateRubric(self):
		opp = self.getChoice('countDown-compOpp')
		if not opp.startswith('>'):
			self.turnOnRubric('loop-incorrectCompOpp')
		
	def renderCode(self):
		v = self.expand('LoopIndex')
		templateVars = {
			'v': v,
			'comp': self.getChoice('countDown-compOpp'),
			'endVal': self.getEndVal(),
			'm': self.expand('MaxToken'),
			'dec': self.expand('Decrement', {'varName':v}),
			'printVar':self.expand('PrintVar', {'varName':v}),
			'varType':self.expand('IntType')
		}
		template = """
		for({varType} {v} = {m}; {v} {comp} {endVal}; {dec}) {{
			{printVar}
		}}
		"""
		return gu.format(template, templateVars)
	
	def getEndVal(self):
		return self.expand('Value', {
			'target': self.getTargetEndValue(),
			'id': 'loop-end-OffByOne'
		})

	# what value should the while loop end on?
	def getTargetEndValue(self):
		opp = self.getChoice('countDown-compOpp')
		# first assume < opp and standard structure
		# note that != opp has the same target as <
		target = 0
		if opp == '>=': target += 1
		return target
