import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# This python class is a RubricSampling Decision
# it generates programs that print the numbers 10 -> 1
class Countdown(Decision):

	def registerChoices(self):
		# these are the main strategies for printing out a
		# countdown

		self.addChoice('loop-style', {
			'for' : 100, 
			'while' : 40, 
			'none' : 1,
			'empty' : 1 
		})

	# we can make some grading choices based on which
	# strategy they chose (did they actually use a loop?)
	def updateRubric(self):
		style = self.getChoice('loop-style')
		hasLoop = style != 'none' and style != 'empty'
		if not hasLoop:
			self.turnOnRubric('doesntUseLoop')

		if style == 'empty':
			self.turnOnRubric('doesntPrintNums')


	# Based on their strategy render a different decision
	def renderCode(self):
		style = self.getChoice('loop-style')
		if style == 'for': return '{ForSoln}'
		if style == 'while': return '{WhileSoln}'
		if style == 'none': return '{NoLoopSoln}'
		if style == 'empty': return ''
