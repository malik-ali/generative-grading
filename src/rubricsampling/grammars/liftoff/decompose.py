import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# given a code bock, either inlines the codeblock,
# or creates a method with the block and invokes it...
# needs to be given an id...
# Problem: lots of ways to chose to decompose AB....
class DecomposablePair(Decision):
	def registerChoices(self):
		self.addChoice('decomposablePair-together', {
			True: 5,
			False: 100
		})

	def renderCode(self):
		raise NotImplementedError('This is currently deprecated')
		pair = self.params['pair']
		assert len(pair) == 2
		decompPair = self.getChoice('decomposablePair-together')
		if decompPair: return self.decompTogether(pair)
		return self.decompSeparate(pair)

	# one method with {ab}. 
	# Both {a} and {b} may be further decompsed
	def decompTogether(self, pair):
		pairBody = ''
		for method in pair:
			pairBody += self.expand('Decompose', {
				'body':method[0],
				'name':method[1],
			})
		pairName = 'pair'
		self.addMethod((pairName, pairBody))
		return pairName + '();'

	# {a} and {b} may be decomposed, but no method {ab}
	def decompSeparate(self, pair):
		ret = ''
		for method in pair:
			ret += self.expand('Decompose', {
				'body':method[0],
				'name':method[1],
			})
		return ret


# Decompose a single action (or not)
# params: name, body
class Decompose(Decision):
	def getKey(self):
		return 'method-' + self.params['name']

	def registerChoices(self):
		raise NotImplementedError('This is currently deprecated')
		self.addChoice(self.getKey(), {
			True: 20,
			False: 100
		})

	def renderCode(self):
		shouldDecompose = self.getChoice(self.getKey())
		if shouldDecompose:
			body = self.params['body']
			name = self.params['name']
			self.addMethod((name, body))
			return name + '();\n'
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