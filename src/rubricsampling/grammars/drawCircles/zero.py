import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


'''
Some students write 0.
Other students write (getWidth() - 10 * diameter) / 2
'''

class ZeroX(Decision):
	def registerChoices(self):
		self.addChoice('zeroX', {
			'0':100,
			'fancy':10
		})

	def preregisterDecisionIds(self):
		return {
			'Diameter': {'diam'}
		}		

	def renderCode(self):
		ch = self.getChoice('zeroX')
		if ch == '0': return ch
		params = {
			'd':self.expand('Diameter', {'id':'diam'}),
			'w':self.expand('Width')
		}
		return gu.format('({w} - 10 * {d}) / 2', params)

class ZeroY(Decision):
	def registerChoices(self):
		self.addChoice('zeroY', {
			'0':100,
			'fancy':10
		})

	def preregisterDecisionIds(self):
		return {
			'Diameter': {'diam'}
		}			

	def renderCode(self):
		ch = self.getChoice('zeroY')
		if ch == '0': return ch
		params = {
			'd':self.expand('Diameter', {'id':'diam'}),
			'h':self.expand('Height')
		}
		return gu.format('({h} - {d}) / 2', params)

class Height(Decision):
	def registerChoices(self):
		self.addChoice('height-style', {
			'getHeight()':100,
			'50':100
		})

	def renderCode(self):
		return self.getChoice('height-style')

class Width(Decision):
	def registerChoices(self):
		self.addChoice('width-style', {
			'getWidth()':100,
			'500':100
		})

	def renderCode(self):
		return self.getChoice('width-style')