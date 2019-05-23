import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


'''
Expected params:
name,
body,
params,
return
'''
class Decompose(Decision):
	def getKey(self):
		return 'method-' + self.params['name']

	def registerChoices(self):
		raise NotImplementedError("Currently deprecated")

		assert 'name' in self.params 
		assert 'body' in self.params 
		assert 'params' in self.params
		assert 'return' in self.params
		# for now, we are only handling null valued methods
		assert self.params['return'] == 'null'

		self.addChoice(self.getKey(), {
			True: 20,
			False: 100
		})

	def renderCode(self):
		shouldDecompose = self.getChoice(self.getKey())
		if shouldDecompose:
			body = self.params['body']
			name = self.params['name']
			argStr = self.getArgumentStr()
			self.addMethod(self.params)
			return '{}({});\n'.format(name, argStr)
		else:
			return self.params['body']


'''
Lets say you want all ways of decomposing
[
{ThingA}
{ThingB}
]

Decompose([{ThingA},{ThingB}])
 --> ThingA can be decomposed
 --> ThingB can be decomposed
 --> Both can be decomposed into the same method

This gets a little out of control when there are > 2 things...
'''

'''

# Maybe decompose out countdown
class CountdownMethod(Decision):
	def registerChoices(self):
		self.addChoice('method-countdown', {
			True: 20,
			False: 100
		})
	def renderCode(self):
		if self.getChoice('method-countdown'):
			body = '{Countdown}'
			name = 'countdown'
			self.addMethod((name, body))
			return name + '();'
		else:
			return '{Countdown}'

# Maybe decompose liftoff
class PrintLiftoffMethod(Decision):
	def registerChoices(self):
		self.addChoice('method-liftoff', {
			True: 20,
			False: 100
		})
	def renderCode(self):
		if self.getChoice('method-liftoff'):
			body = '{PrintLiftoff}'
			name = self.expand('LiftoffMethodName')
			self.addMethod((name, body))
			return name + '();'
		else:
			return '{PrintLiftoff}'

'''