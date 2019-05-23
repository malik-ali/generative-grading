import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: IncrementX
# ------------------------
class IncrementX(Decision):
	def registerChoices(self):
		self.addChoice('incrementStyle', {
			'+=': 100,
			'full': 20,
			'noAdd':10
		})

	def updateRubric(self):
		if self.getChoice('incrementStyle') == 'noAdd':
			self.turnOnRubric('incorrectUpdate')

	def renderCode(self):
		d = self.expand('IncrementDelta')
		style = self.getChoice('incrementStyle')

		if style == '+=': return 'x += ' + d
		if style == 'full': return 'x = x + ' + d
		if style == 'noAdd': return 'x = ' + d


class IncrementDelta(Decision):
	def registerChoices(self):
		self.addChoice('incrementeDelta', {
			'diameter':100,
			'i*d':100
		})

	def updateRubric(self):
		if self.getChoice('incrementeDelta') == 'i*d':
			self.turnOnRubric('incorrectUpdate')

	def preregisterDecisionIds(self):
		return {
			'Mult': {'mult-incDelta'},
			'Diameter': {'delta'}
		}	

	def renderCode(self):
		d = self.expand('Diameter', {'id':'delta'})
		style = self.getChoice('incrementeDelta')
		if style == 'diameter': return d

		return self.expand('Mult', {'a': 'i', 'b':d,'id':'mult-incDelta'})

