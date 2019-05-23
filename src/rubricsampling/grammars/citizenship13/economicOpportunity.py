import sys
sys.path.insert(0, '../../')

import generatorUtils as gu
import random
from base import Decision


class EconomicOpportunity(Decision):

	def registerChoices(self):
		self.addChoice('econVerb', {
			'gain' : 50,
			'obtain' : 50,
			'practice' : 50,
			'find' : 50,
			'pursue':50,
			'seek':50,
			'search for':50
		})
		self.addChoice('econNoun', {
			'economic opportunity': 100,
			'economic prospects': 100,
			'economic hope': 10,
			'economic possibilities': 10,
		})
		self.addChoice('econQualifier', {
			'':100,
			'more':10,
			'better ':20,
		})

	def renderCode(self):
		noun = self.getChoice('econNoun')
		qualifier = self.getChoice('econQualifier')
		optUs = self.expand('OptUS')
		
		nounPhrase = qualifier + noun + optUs
		self.addOrSetState('noun', nounPhrase)
		self.addOrSetState('verb', self.getChoice('econVerb'))
		self.addOrSetState('nounAloneOk', True)
		return ''