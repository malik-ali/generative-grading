import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

reusableDecisions = {
	'BinaryOpp':2,
	'MoveTurn':2, 
	'BodyWeeds':2,
	'MainMove':2,
	'ConstMove':3, 
	'MoveVar':2, 
	'MoveConst':3, 
	'MoveAmt':2,
	'MoveDir':2, 
	'MultCoefficient':2,
	'MainTurn':2, 
	'TurnVar':2, 
	'TurnConst':2, 
	'TurnAmt':2,
	'TurnDir':2, 
	'TurnXKCoefficient':2,
	'AddCoefficient':2, 
	'DivideCoefficient':2,
	'ForHeaderPixels':2
}

'''
Mistakes:
'''


'''
Some things:

Put in an ability prior

Handle the "work in progress" concept
'''
class Program(Decision):

	def preregisterDecisionIds(self):
		return self.getAllResuableDecisions()

	def renderCode(self):
		# only recognize errors from your first move/turn
		self.addState('firstMove', True)
		self.addState('firstTurn', True)
		self.addResuableCounts()
		ans = self.expand('ShapeFor')
		self.updateRubricForMissingOpps(ans)
		return ans

	def updateRubricForMissingOpps(self,ans):
		if not 'Move' in ans:
			self.turnOnRubric('Move: no move')
		if not 'Turn' in ans:
			self.turnOnRubric('Turn: no turn')
		if not 'For(' in ans:
			self.turnOnRubric('For loop: no loop')
		if not 'Repeat' in ans:
			self.turnOnRubric('Single shape: missing repeat')

	##############

	'''
	Special code to preregister all reusable decisions
	And to initialize a count variable which keeps track of
	Which index we are on
	'''

	def getAllResuableDecisions(self):
		decisions = {}
		for d in reusableDecisions:
			m = reusableDecisions[d]
			decisions[d] = [
				d+'_{}'.format(i) for i in range(m)
			]
		return decisions

	def addResuableCounts(self):
		for d in reusableDecisions:
			self.addState(d +'_count', 0)