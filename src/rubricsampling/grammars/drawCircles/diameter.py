import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import ReusableDecision

'''
WARNING: 
How do we allow for this decision to be made
differenently in different contexts? How does
this impact our ability to rationalize about
'''
class Diameter(ReusableDecision):

	def registerChoices(self):

		# check if this decision was already made
		if self.hasChoice(self.getThoughtKey()):
			return

		# how do they think about the diameter?
		self.addChoice(self.getThoughtKey(), {
			'standard':100,
			'confuseDiamRadius':10,
			'twoTimesRadius':30,
			'getHeight':30
		}, is_prefix=True)

		# where do they declare the variable?
		self.addChoice(self.getStyleKey(), {
			'magicNumber': 100,
			'const': 60,
			'localVar':40,
			'instanceVar':1
		}, is_prefix=True)

	def updateRubric(self):
		thought = self.getChoice(self.getThoughtKey())
		if thought == 'confuseDiamRadius':
			self.turnOnRubric('confuseDiamRadius')

	def renderCode(self):
		thought = self.getChoice(self.getThoughtKey())

		if thought == 'standard': return self.getStandard()
		if thought == 'confuseDiamRadius': return self.getRadius()
		if thought == 'twoTimesRadius': return self.getTwoRadius()
		if thought == 'getHeight': return 'getHeight()'

	def getThoughtKey(self):
		prefix = self.params['id']
		return prefix + '-diam-thought'

	def getStyleKey(self):
		prefix = self.params['id']
		return prefix + '-diam-style'

	def getStandard(self):
		style = self.getChoice(self.getStyleKey())
		if style == 'magicNumber': return '50'
		return self.getDiamVar()

	def getRadius(self):
		style = self.getChoice(self.getStyleKey())
		if style == 'magicNumber': return '25'
		return self.getRadiusVar()

	def preregisterDecisionIds(self):
		return {
			'Mult': {'mult-2r'},
			'IntType': {'diamVarType', 'radiusVarType'}
		}

	def getTwoRadius(self):
		style = self.getChoice(self.getStyleKey())
		r = '25'
		if style != 'magicNumber': 
			r = self.getRadiusVar()
		return self.expand('Mult', {'a':2, 'b':r, 'id':'mult-2r'})

	# The variable declaration parts
	def getDiamVar(self):
		style = self.getChoice(self.getStyleKey())
		if style == 'const':
			if not 'DIAMETER' in self.getState('constants'):
				varType = self.expand('IntType', {'id':'diamVarType'})
				line = 'private static final {} DIAMETER = 50;'.format(varType)
				self.getState('constants')['DIAMETER'] = line
			return 'DIAMETER'
		if style == 'localVar': 
			if not 'diam' in self.getState('loop-init-vars'):
				varType = self.expand('IntType', {'id':'diamVarType'})
				line = '{} diam = 50;'.format(varType)
				self.getState('loop-init-vars')['diam'] = line
			return 'diam'
		if style == 'instanceVar':
			return 'diam'

	def getRadiusVar(self):
		style = self.getChoice(self.getStyleKey())
		if style == 'const':
			if not 'RADIUS' in self.getState('constants'):
				varType = self.expand('IntType', {'id': 'radiusVarType'})
				line = 'private static final {} RADIUS = 25;'.format(varType)
				self.getState('constants')['RADIUS'] = line
			return 'RADIUS'
		if style == 'localVar': 
			if not 'radius' in self.getState('loop-init-vars'):
				varType = self.expand('IntType', {'id': 'radiusVarType'})
				line = '{} radius = 25;'.format(varType)
				self.getState('loop-init-vars')['radius'] = line
			return 'radius'
		if style == 'instanceVar':
			return 'radius'



