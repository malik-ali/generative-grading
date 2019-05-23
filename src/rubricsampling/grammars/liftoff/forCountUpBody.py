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
	def registerChoices(self):
		# do they combine printing and calculation
		self.addChoice('countUp-bodyStrategy', {
			'inline': 50,
			'decrementCounter': 50
		})

	def renderCode(self):
		strategy = self.getChoice('countUp-bodyStrategy')
		if strategy == 'inline': return '{ForUpInline}'
		if strategy == 'decrementCounter': return '{ForUpDecement}'


'''
# decrementCounter
int x = 10;
for(int i = 0; i < 10; i++) {
	println(x);
	x -= 1;
}

# TODO
# rare decrement (not implemented yet)
int x = 11;
for(int i = 0; i < 10; i++) {
	x -= 1;
	println(x);
}

# TODO
# rare not sure if this shows up
int x = 10;
for(int i = 0; i < 10; i++) {
	println(i--)
}
'''

class ForUpDecement(Decision):
	def renderCode(self):
		# since we anonimize, I don't spend too much time on vars
		v = 'x'
		# save the var into forloops-pre-vars 
		# that way it can be defined before the loop
		tempInitValue = self.expand('MaxToken')
		tempDeclaration = 'int {} = {};'.format(v, tempInitValue)
		self.getState('forloop-pre-vars').append(tempDeclaration)

		templateVars = {
			'printVar': self.expand('PrintVar', {'varName':v}),
			'decement': self.expand('Decrement', {'varName':v}),
		}
		template = '''
		{printVar}
		{decement};
		'''
		return gu.format(template, templateVars)

'''
# inline strategy
for(int i = 0; i < 10; i++) {
	println(10 - i);
}

# inline-temp strategy
for(int i = 0; i < 10; i++) {
	int temp = 10 - i;
	println(temp);
}

# inline-predefTemp strategy
# TODO, make more init choices
int temp = 0;
for(int i = 0; i < 10; i++) {
	temp = 10 - i;
	println(temp)
}
'''

class ForUpInline(Decision):
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
		return 'println({});'.format(term)

	def innerTemp(self):
		tempVars = {
			'temp':'x',
			'calc': self.expand('ForUpInlineCalc')
		}

		template = '''
		int {temp} = {calc};
		println({temp});
		'''
		return gu.format(template, tempVars)

	def outerTemp(self):
		# add a preloop var definition
		self.getState('forloop-pre-vars').append('int x = 0;')
		tempVars = {
			'temp':'temp',
			'calc': self.expand('ForUpInlineCalc')
		}
		template = '''
		{temp} = {calc};
		println({temp})
		'''
		return gu.format(template, tempVars)


# ie 10 - i
class ForUpInlineCalc(Decision):
	def renderCode(self):
		maxToken = self.expand('MaxToken')
		i = self.getState('forloop-i')
		return '{} - {}'.format(maxToken, i)
