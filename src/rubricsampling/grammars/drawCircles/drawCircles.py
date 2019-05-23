import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class DrawCircles(Decision):

	def registerChoices(self):
		self.addChoice('loop-style', {
			'for' : 100, 
			'nonLoop' : 2,
		})

	def renderCode(self):
		self.addState('loop-init-vars', {})
		self.addState('loop-pre-vars', [])
		style = self.getChoice('loop-style')

		params = {
			'body': self.renderBody(style),
			'preLoopVars': self.renderPreLoopVars(),
			'initVars':self.renderInitVars()
		}
		template ='''
			{initVars}
			{preLoopVars}
			{body}
		'''
		return gu.format(template, params)
		
	def renderBody(self, style):
		if style == 'for': return self.expand('ForSoln')
		if style == 'while': return self.expand('WhileSoln')
		if style == 'nonLoop': return self.expand('NonLoopSoln')

	def renderPreLoopVars(self):
		lines = self.getState('loop-pre-vars')
		if lines == None: return ''
		return self.expand('PermuteLines', {'lines':lines,'id':'pre-loop'})

	def renderInitVars(self):
		constMap = self.getState('loop-init-vars')
		if constMap == None: return ''
		lines = []
		for key in constMap:
			lines.append(constMap[key])
		return self.expand('PermuteLines', {'lines': lines, 'id':'init'})
