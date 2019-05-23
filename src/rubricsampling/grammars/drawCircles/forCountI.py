import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Count Up
# ------------------------
# This is for people who have a for loop with ++ as the
# for loop operator. 

class ForCountI(Decision):
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

	def preregisterDecisionIds(self):
		return {
			'ForIntType': {'loopStartType'}
		}
	

	def renderCode(self):
		# this has to be precalculated so it can be used in other steps
		v = self.expand('LoopIndex')
		self.addState('forloop-i', v)

		templateVars = {
			'v' : v,
			'comp': self.getChoice('countUp-compOpp'),
			'endVal' : self.expand('ForUpEndVal'),
			'inc' : self.expand('Increment', {'varName':v}),
			'varType' : self.expand('ForIntType', {'id': 'loopStartType'}),
			# this non terminal defines the strategy of using the loop
			'forCountUpBody' : self.expand('ForCountUpBody'),
		}

		template = """
		for({varType} {v} = 0; {v} {comp} {endVal}; {inc}) {{
			{forCountUpBody}
		}}
		"""
		return gu.format(template, templateVars)


'''
at the moment this end val has assumed a lot of complexity
'''
class ForUpEndVal(Decision):
	def registerChoices(self):
		# if they use START instead of a magic number
		self.addChoice('loopEnd-useConst', {
			True: 100,
			False: 100
		})

		# are they off by one on last value
		self.addChoice('loop-endValue', {
			'tooFew': 1, # the last number printed is a 2
			'tooMany': 20,  # the last number printed is a 0 
			'correctNum' : 100
		})

	def updateRubric(self):
		if self.getChoice('loop-endValue') != 'correctNum':
			self.turnOnRubric('loop-endOffByOne')


	def renderCode(self):
		useConst = self.getChoice('loopEnd-useConst')
		opp = self.getChoice('countUp-compOpp')
		offByOne = self.getChoice('loop-endValue')

		if useConst: return self.withConst(opp, offByOne)
		else: return self.withoutConst(opp, offByOne)
		
	def withConst(self, opp, offByOne):
		const = 'N_CIRCLES'
		self.addMaxConst(const)
		# perfect
		if offByOne == 'correctNum':
			if opp == '<=': return const + ' - 1'
			return const
		# print "0"
		if offByOne == 'tooMany':
			if opp == '<=': return const
			return const + ' + 1'
		# don't print "1"
		if offByOne == 'tooFew':
			if opp == '<=': return const + '- 2'
			return const + ' - 1'



	def withoutConst(self, opp, offByOne):
		val = 10
		# start by assuming opp is <
		if offByOne == 'correctNum': 	val = 10
		if offByOne == 'tooMany': val = 11
		if offByOne == 'tooFew':  val = 9

		# if its actually <= subtract one
		if opp == '<=': val -= 1
		return str(val)

	def preregisterDecisionIds(self):
		return {
			'IntType': {'maxConstType'}
		}

	def addMaxConst(self, name):
		varType = self.expand('IntType', {'id':'maxConstType'})
		line = 'private static final {} {} = 10;'.format(varType, name)
		self.getState('constants')[name] = line
