import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: While Loop Soln
# -------------------------
# For students who used a while loop to count down.
class WhileSoln(Decision):

	def registerChoices(self):
		# does the student use >= or > in the for loop
		self.addChoice('while-compOpp', {
			'>=' : 100, 
			'>' : 100,
			'<': 1,
			'<=': 1,
			'!=':10
		})

		self.addChoice('while-struct', {
			'standard':100,
			'decPrint':5
		})

	# some comp operators are going to be very wrong
	def updateRubric(self):
		opp = self.getChoice('while-compOpp')

		# This while solution only accounts for a decrementing while loop.
		# Comparing to >, >=, or != should be fine.
		if opp.startswith('<'):
			self.turnOnRubric('loop-incorrectCompOpp')


	def renderCode(self):
		struct = self.getChoice('while-struct')
		if struct == 'standard': return '{WhileStandard}'
		if struct == 'decPrint': return '{WhileDecPrint}'


'''
int x = START;
while(x >= 1) {
	println(x);
	x--;
}
int x = START;
while(x > 0) {
	println(x);
	x--;
}
int x = START;
while(x != 0) {
	println(x);
	x--;
}
'''
class WhileStandard(Decision):

	def preregisterDecisionIds(self):
		return {
			'Value': {'loop-end-OffByOne'}
		}

	# the most common while loop
	def renderCode(self):
		v = self.expand('LoopIndex')
		templateVars = {
			'v' : v,
			'comp':self.getChoice('while-compOpp'),
			'm': self.expand('MaxToken'),
			'endValue':self.getEndValue(),
			'dec' : self.expand('Decrement', {'varName':v}),
			'printVar': self.expand('PrintVar', {'varName':v}),
			'varType':self.expand('IntType')
		}
		template = '''
		{varType} {v} = {m};
		while({v} {comp} {endValue}) {{
			{printVar}
			{dec};
		}}
		'''
		return gu.format(template, templateVars)


	# what is the end value for your loop?
	def getEndValue(self):
		return self.expand('Value', {
			'target': self.getTargetEndValue(),
			'id': 'loop-end-OffByOne'
		})

	# what value should the while loop end on?
	def getTargetEndValue(self):
		opp = self.getChoice('while-compOpp')
		# first assume < opp and standard structure
		# note that != opp has the same target as <
		target = 0
		if opp == '>=': target += 1
		return target

'''
int x = START;
println(x);
while(x > 1) {
	x--;
	println(x);
}

int x = START;
println(START);
while(x > 1) {
	x--;
	println(x);
}

println(START);
int x = START;
while(x > 1) {
	x--;
	println(x);
}
'''
class WhileDecPrint(Decision):

	def registerChoices(self):
		self.addChoice('whileDecPrint-setup', {
			'printVal': 100,
			'printStart':50,
			'prePrintStart':50
		})

	def preregisterDecisionIds(self):
		return {
			'Value': {'loop-end-OffByOne'}
		}

	def renderCode(self):
		v = self.expand('LoopIndex')
		printVar = self.expand('PrintVar', {'varName':v})
		templateVars = {
			'v' : v,
			'comp':self.getChoice('while-compOpp'),
			'endValue':self.getEndValue(),
			'dec' : self.expand('Decrement', {'varName':v}),
			'setup': self.getSetup(v, printVar),
			'printVar': printVar
		}
		template = '''
		{setup}
		while({v} {comp} {endValue}) {{
			{dec};
			{printVar}
		}}
		'''
		return gu.format(template, templateVars)

	def getSetup(self, v, printVar):
		templateVars = {
			'v':v,
			'varType':self.expand('IntType'),
			'm': self.expand('MaxToken'),
			'printVar':printVar
		}
		
		# switch on the three structures
		struct = self.getChoice('whileDecPrint-setup')
		if struct == 'printVal':
			return gu.format('''
				{varType} {v} = {m};
				{printVar}
			''', templateVars)
		if struct == 'printStart':
			return gu.format('''
				{varType} {v} = {m};
				println(START);
			''', templateVars)
		if struct == 'prePrintStart':
			return gu.format('''
				println(START);
				{varType} {v} = {m};
			''', templateVars)
		raise Exception('unknown stuct ' + struct)

	# what is the end value for your loop?
	def getEndValue(self):
		return self.expand('Value', {
			'target':self.getTargetEndValue(),
			'id': 'loop-end-OffByOne'
		})

	# what value should the while loop end on?
	def getTargetEndValue(self):
		opp = self.getChoice('while-compOpp')
		# first assume < opp and standard structure
		# note that != opp has the same target as <
		target = 1
		if opp == '>=': target += 1
		return target

