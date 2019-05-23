import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class DrawCircleXY(Decision):
	def renderCode(self):
		# first make all the body choices, *then*
		# decide if you want to decompose. Body choices
		# impact which parameters are necessary
		return self.expand('DrawCircleXYBody', self.params)

class DrawCircleXYBody(Decision):

	def registerChoices(self):
		# this choice is basically a rubric item..
		self.addChoice('adds-same-oval',{
			False: 100,
			True: 100
		})

	def updateRubric(self):
		if self.getChoice('adds-same-oval'):
			self.turnOnRubric('adds-same-oval')

	def renderCode(self):
		# some students forget to create a new goval each
		# time. They declare the goval in the loop-pre-vars
		if self.getChoice('adds-same-oval'):
			return self.renderAddsSame()
		return self.expand('RenderAddsNewGOval', {
			'x':self.params['x']
		})
	
	def preregisterDecisionIds(self):
		return {
			'Diameter': {'diam'}
		}

	def renderAddsSame(self):
		tempVars = {
			'd':self.expand('Diameter', {'id':'diam'}),
			'v': 'oval',
			'x': self.params['x'],
			'y': self.expand('Y')
		}
		preLine = gu.format('GOval {v} = new GOval({d}, {d});', tempVars)
		addLine = gu.format('add({v}, {x}, {y});', tempVars)
		self.getState('loop-pre-vars').append(preLine)
		return addLine


class RenderAddsNewGOval(Decision):
	def registerChoices(self):
		self.addChoice('goval-constructor', {
			'widthHeight':100,
			'xyWidthHeight':10
		})

	def renderCode(self):
		constructor = self.getChoice('goval-constructor')
		if constructor == 'widthHeight':
			return self.renderStandard()
		else:
			return self.renderXYWH()

	def preregisterDecisionIds(self):
		return {
			'Diameter': {'diam'}
		}	

	def renderXYWH(self):
		x = self.params['x']
		tempVars = {
			'd':self.expand('Diameter', {'id':'diam'}),
			'v': 'oval',
			'x': x,
			'y': self.expand('Y')
		}

		preLine = gu.format('GOval {v} = new GOval({x}, {y}, {d}, {d});', tempVars)
		addLine = gu.format('add({v});', tempVars)
		self.getState('loop-start-lines').append(preLine)
		return addLine

	def renderStandard(self):
		x = self.params['x']
		tempVars = {
			'd':self.expand('Diameter', {'id':'diam'}),
			'v': 'oval',
			'x': x,
			'y': self.expand('Y')
		}

		preLine = gu.format('GOval {v} = new GOval({d}, {d});', tempVars)
		addLine = gu.format('add({v}, {x}, {y});', tempVars)
		self.getState('loop-start-lines').append(preLine)
		return addLine

# circles that are drawn outside a loop
# each need their own var name...
class DrawCircleNoLoop(Decision):

	def registerChoices(self):
		pass

	def preregisterDecisionIds(self):
		return {
			'Diameter': {'diam'}
		}		

	def renderCode(self):
		i = self.params['i']
		x = i * 50
		templateVars = {
			'd': self.expand('Diameter', {'id':'diam'}),
			'v': 'oval' + str(i),
			'x': x,
			'y': '0'
		}

		return '''
		GOval {v} = new GOval({d}, {d});
		add({v}, {x}, {y});
		'''.format(**templateVars)