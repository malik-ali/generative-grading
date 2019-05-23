import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

# Decision: Print Liftoff
# ------------------------
# You are supposed to print liftoff at the end of 
# countdown
class PrintLiftoff(Decision):
	def registerChoices(self):
		self.addChoice('printLiftoff', {
			True : 100, # they do print liftoff
			False : 5   # they forget to print liftoff
		})

	def updateRubric(self):
		printsLiftoff = self.getChoice('printLiftoff')
		if not printsLiftoff:
			self.turnOnRubric('noPrintLiftoff')

	def renderCode(self):
		if self.getChoice('printLiftoff'):
			return '{Liftoff}'
		else:
			return ''

# Decision: Liftoff
# ------------------------
# There are lots of ways of writing liftoff
class Liftoff(Decision):
	def registerChoices(self):
		self.addChoice('liftoff-printMethod', {
			'println':150,
			'print':1
		})
		self.addChoice('liftoff-stringBase', {
			'liftoff':5,
			'Liftoff':100,
			'LIFTOFF':5,
			'Lift off': 5,
			'LiftOff':5,
			'Lift off':5,
			'Liftoff ':5
		})
		self.addChoice('liftoff-stringExcitement', {
			'':100,
			'!':5,
			'!!!':1
		})

	def renderCode(self):
		templateVars = {
			'method':self.getChoice('liftoff-printMethod'),
			'base':self.getChoice('liftoff-stringBase'),
			'excite':self.getChoice('liftoff-stringExcitement')
		}
		template = '{method}("{base}{excite}");'
		return gu.format(template, templateVars)