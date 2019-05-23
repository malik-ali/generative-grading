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
			'>=' : 1, 
			'>' : 1,
			'<': 100,
			'<=': 100,
			'!=':10
		})

		self.addChoice('while-struct', {
			'standard':100,
			'incPrint':5
		})

	# some comp operators are going to be very wrong
	def updateRubric(self):
		opp = self.getChoice('while-compOpp')
		if opp.startswith('>'):
			self.turnOnRubric('loop-incorrectCompOpp')

	def renderCode(self):
		raise Exception('not updated')
		struct = self.getChoice('while-struct')
		if struct == 'standard': return '{WhileStandard}'
		if struct == 'incPrint': return '{WhileIncPrint}'


'''
int x = 0;
while(x < 500) {
	drawCircle(x);
	x += 50;
}
int i = 0;
while(i < 10) {
	drawCircle(i * 50);
	i++;
}
int x = 0;
while(x != 500) {
	drawCircle(x);
	x += 50;
}
'''
class WhileStandard(Decision):

	def preregisterDecisionIds(self):
		return {
			'Value': {'rubric-loop-end-offByOne'},
			'IntType': {'loopStartType'}
		}

	# the most common while loop
	def renderCode(self):
		v = self.expand('LoopIndex')
		templateVars = {
			'v' : v,
			'comp':self.getChoice('while-compOpp'),
			'start': self.expand('StartToken'),
			'endValue':self.getEndValue(),
			'dec' : self.expand('Decrement', {'varName':v}),
			'printVar': self.expand('PrintVar', {'varName':v}),
			'varType':self.expand('IntType', {'id':'loopStartType'})
		}
		template = '''
		{varType} {v} = {start};
		while({v} {comp} {endValue}) {{
			{printVar}
			{dec};
		}}
		'''
		return gu.format(template, templateVars)
		
	# what is the end value for your loop?
	def getEndValue(self):
		return self.expand('Value', {
			'target':self.getTargetEndValue(),
			'key':'rubric-loop-end-offByOne'
		})

	# what value should the while loop end on?
	def getTargetEndValue(self):
		opp = self.getChoice('while-compOpp')
		# first assume < opp and standard structure
		# note that != opp has the same target as <
		target = 10
		if opp == '<=': target -= 1
		return target

'''
int x = 0;
drawCircle(x);
while(x < 500) {
	x+=50;
	drawCircle(x);
}

int i = 0;
drawCircle()
'''
class WhileIncPrint(Decision):

	def registerChoices(self):
		self.addChoice('whileDecPrint-setup', {
			'printVal': 100,
			'printStart':50,
			'prePrintStart':50
		})

	def preregisterDecisionIds(self):
		return {
			'Value': {'rubric-loop-end-offByOne'},
			'IntType': {'loopStartType'}
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
			'varType':self.expand('IntType', {'id':'loopStartType'}),
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
			'key':'rubric-loop-end-offByOne'
		})

	# what value should the while loop end on?
	def getTargetEndValue(self):
		opp = self.getChoice('while-compOpp')
		# first assume < opp and standard structure
		# note that != opp has the same target as <
		target = 1
		if opp == '>=': target += 1
		return target

