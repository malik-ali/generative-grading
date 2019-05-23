import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class OptTheir(Decision):
	def registerChoices(self):
		self.addChoice('hasTheir', {
			'True': 20,
			'False':40
		})

	def renderCode(self):
		if(self.getChoice('hasTheir') == 'False'):
			return ''
		else:
		 	return 'their '

class OptHad(Decision):
	def registerChoices(self):
		self.addChoice('hasHad', {
			'True': 20,
			'False':40
		})

	def renderCode(self):
		if(self.getChoice('hasHad') == 'False'):
			return ''
		else:
		 	return 'had '

class OptFor(Decision):
	def registerChoices(self):
		self.addChoice('hasFor', {
			'True': 20,
			'False':40
		})

	def renderCode(self):
		if(self.getChoice('hasFor') == 'False'):
			return ''
		else:
		 	return 'for '

class OptAs(Decision):
	def registerChoices(self):
		self.addChoice('hasAs', {
			'True': 20,
			'False':40
		})

	def renderCode(self):
		if(self.getChoice('hasAs') == 'False'):
			return ''
		else:
		 	return 'as '


class OptA(Decision):
	def registerChoices(self):
		self.addChoice('hasA', {
			'True': 20,
			'False':40
		})

	def renderCode(self):
		if(self.getChoice('hasA') == 'False'):
			return ''
		else:
		 	return 'a '
