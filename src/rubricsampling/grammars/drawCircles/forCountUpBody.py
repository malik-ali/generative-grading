import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: For Loop Body
# ------------------------
# This is for people who have a count up for loop 
# and calculate the output. 
# Important: there is an option to add vars that
#   should be predefined before the loop.
# The main strategies are to use a temp var that decrements
# or to recalculate the next countdown each iteration

class ForCountUpBody (Decision):
	def getKey(self):
		return 'countUp-bodyStrategy'

	def registerChoices(self):
		# do they combine printing and calculation
		self.addChoice(self.getKey(), {
			'ForUpCalcX': 50,
			'ForUpIncrementX': 50
		})

	def renderCode(self):
		stragegyNonTerminal = self.getChoice(self.getKey())
		stateKey = 'loop-start-lines'
		# the lines at the start of the for loop can be permuted
		self.addState(stateKey, [])
		tempVars = {
			'strategy': self.expand(stragegyNonTerminal),
			'forLoopStartLines': self.expand('PermuteLines', {
				'lines': self.getState(stateKey), 
				'id':stateKey
			})
		}
		return gu.format('''
			{forLoopStartLines}
			{strategy}
		''', tempVars)


'''
# ForUpIncrementX
int x = 0;
for(int i = 0; i < 10; i++) {
	drawCircle(x, 0);
	x += 50;
}
'''

class ForUpIncrementX(Decision):
	def renderCode(self):
		# since we anonimize, I don't spend too much time on vars
		v = 'x'
		# save the var into loop-pre-vars 
		# that way it can be defined before the loop
		tempInitValue = self.expand('ZeroX')
		tempDeclaration = 'int {} = {};'.format(v, tempInitValue)
		self.getState('loop-pre-vars').append(tempDeclaration)

		templateVars = {
			'drawCircle': self.expand('DrawCircleXY', {'x':'x'}),
			'increment': self.expand('IncrementX', {'varName':v}),
		}
		template = '''
		{drawCircle}
		{increment};
		'''
		return gu.format(template, templateVars)

'''
# inline strategy
for(int i = 0; i < 10; i++) {
	drawCircle(i * 50);
}

# inline-temp strategy
for(int i = 0; i < 10; i++) {
	int x = i * 50;
	drawCircle(x);
}

# inline-predefTemp strategy
# TODO, make more init choices
int x = 0;
for(int i = 0; i < 10; i++) {
	x = i * 50;
	drawCircle(x);
}
'''

class ForUpCalcX(Decision):
	def registerChoices(self):
		self.addChoice('countUp-inlineStrategy', {
			'noTemp': 100,
			'innerTemp': 50,
			'outerTemp': 5
		})

	def renderCode(self):
		strategy = self.getChoice('countUp-inlineStrategy')
		if strategy == 'noTemp': return self.noTemp()
		if strategy == 'innerTemp': return self.innerTemp()
		if strategy == 'outerTemp': return self.outerTemp()
		raise Exception('unknown strategy')

	def noTemp(self):
		term = self.expand('ForUpInlineCalc')
		return self.expand('DrawCircleXY', {'x':term})

	def innerTemp(self):
		tempVars = {
			'temp':'x',
			'calc': self.expand('ForUpInlineCalc')
		}

		line = gu.format('int {temp} = {calc};', tempVars)
		self.getState('loop-start-lines').append(line)
		ret = self.expand('DrawCircleXY', {'x':'x'})
		return ret

	def outerTemp(self):
		# add a preloop var definition
		tempVars = {
			'temp':'x',
			'calc': self.expand('ForUpInlineCalc')
		}
		line = gu.format('{temp} = {calc};', tempVars)
		self.getState('loop-start-lines').append(line)
		self.getState('loop-pre-vars').append('int x = {ZeroX};')

		return self.expand('DrawCircleXY', {'x':'x'})


# ie i * 50 or 50 * i
class ForUpInlineCalc(Decision):
	def preregisterDecisionIds(self):
		return {
			'Mult': {'mult-forInlineCalc'},
			'Diameter': {'delta'}
		}

	def renderCode(self):
		d = self.expand('Diameter', {'id':'delta'})
		return self.expand('Mult', {'a':'i', 'b':d, 'id':'mult-forInlineCalc'})
