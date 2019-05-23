import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Count Up
# ------------------------
# This is for people who have a for loop with ++ as the
# for loop operator.

class ForCountUp(Decision):
	def registerChoices(self):
		# does the student use >= or > in the for loop
		self.addChoice('countUp-compOpp', {
			'>=' : 1,
			'>' : 1,
			'<' : 200,
			'<=' : 5,
			'!=': 5
		})


	def updateRubric(self):
		# if they are counting up, must use "<" or "<="
		opp = self.getChoice('countUp-compOpp')
		if opp.startswith('>'):
			self.turnOnRubric('loop-incorrectCompOpp')

	def renderCode(self):
		# keep track of variables declared before the loop
		self.addState('forloop-pre-vars', [])

		# this has to be precalculated so it can be used in other steps
		v = self.expand('LoopIndex')
		self.addState('forloop-i', v)



		templateVars = {
			'v' : v,
			'comp': self.getChoice('countUp-compOpp'),
			'endVal' : self.expand('ForUpEndVal'),
			'inc' : self.expand('Increment', {'varName':v}),
			'varType' : self.expand('IntType'),
			# this non terminal defines the strategy of using the loop
			'forCountUpBody' : self.expand('ForCountUpBody'),
			'predeclardVars': self.renderPredeclaredVars()
		}

		template = """
		{predeclardVars}
		for({varType} {v} = 0; {v} {comp} {endVal}; {inc}) {{
			{forCountUpBody}
		}}
		"""
		return gu.format(template, templateVars)

	# return all the variables which were added to
	# forloop-pre-vars
	def renderPredeclaredVars(self):
		ret = ''
		for line in self.getState('forloop-pre-vars'):
			ret += line + '\n'
		return ret


'''
at the moment this end val has assumed a lot of complexity
'''
class ForUpEndVal(Decision):
	def registerChoices(self):
		# if they use START instead of a magic number
		self.addChoice('loopEnd-useConst', {
			True: 100,
			False: 50
		})

		# are they off by one on last value
		self.addChoice('forUploop-endOffByOne', {
			'endPrint2': 1, # the last number printed is a 2
			'endPrint0': 20,  # the last number printed is a 0
			False : 100
		})

	def updateRubric(self):
		offByOne = self.getChoice('forUploop-endOffByOne')
		if offByOne != False:
			self.turnOnRubric('forUploop-endOffByOne')

	def renderCode(self):
		useConst = self.getChoice('loopEnd-useConst')
		opp = self.getChoice('countUp-compOpp')
		offByOne = self.getChoice('forUploop-endOffByOne')

		if useConst: return self.withConst(opp, offByOne)
		else: return self.withoutConst(opp, offByOne)

	def withConst(self, opp, offByOne):
		# perfect
		if offByOne == False:
			if opp == '<=': return '{ConstName} - 1'
			return '{ConstName}'
		# print "0"
		if offByOne == 'endPrint0':
			if opp == '<=': return '{ConstName}'
			return '{ConstName} + 1'
		# don't print "1"
		if offByOne == 'endPrint2':
			if opp == '<=': return '{ConstName} - 2'
			return '{ConstName} - 1'
		print(offByOne, type(offByOne))
		print(opp)
		raise Exception('unknown rvs')

	def withoutConst(self, opp, offByOne):
		val = 10
		# start by assuming opp is <
		if offByOne == False: 		val = 10
		if offByOne == 'endPrint0': val = 11
		if offByOne == 'endPrint2': val = 9

		# if its actually <= subtract one
		if opp == '<=': val -= 1
		return str(val)
