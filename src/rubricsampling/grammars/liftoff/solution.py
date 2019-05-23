import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision

class Solution(Decision):
	def renderCode(self):

		templateVars = {
			'Countdown': self.expand('Countdown'),
			'PrintLiftoff': self.expand('PrintLiftoff')
		}

		template = '''
			{Countdown}
			{PrintLiftoff}
		'''

		return gu.format(template, templateVars)
		
