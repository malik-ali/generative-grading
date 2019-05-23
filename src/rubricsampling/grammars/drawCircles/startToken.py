import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
from base import Decision


class StartToken(Decision):

	def renderCode(self):
		return '0'