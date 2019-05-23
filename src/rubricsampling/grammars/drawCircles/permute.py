import sys
sys.path.insert(0, '../..')

import generatorUtils as gu
import random
import copy
from base import Decision

'''
INSIGHT:
The idea here is to use index-strings as the choices.
For all permutations we create a unique index string eg:
"1203" or "0123"
Those are the choices (equally weighted)
and we turn them into int-indecies when we want to 
reconstruct the permutation
'''
# For now, not using permutations
class PermuteLines(Decision):
	# def registerChoices(self):
		# idStr = self.params['id']
		# assert idStr != None
		# # for now we only allow up to 10 elements in permute
		# # note: 10! = 3.6 million
		# if self.getN() >= 10:
		# 	raise Exception('Too many elements in permute')

		# self.addChoice(self.getKey(), self.getChoices())

	def renderCode(self):
		return '\n'.join(self.params['lines'])
		# permuteStr = self.getChoice(self.getKey())
		# lines = self.params['lines']

		# ret = ''
		# for ch in permuteStr:
		# 	i = int(ch)
		# 	ret += lines[i] + '\n'
		# return ret	

	def getN(self):
		return len(self.params['lines'])

	def getKey(self):
		return 'permute-' + self.params['id']

	# make all permutation strings. Most of the work
	# is done by the recursive getChoicesHelper method
	def getChoices(self):
		allPerms = self.getChoicesHelper([])
		ret = {}
		for perm in allPerms:
			permStr = ''
			for v in perm:
				permStr += str(v)
			ret[permStr] = 1
		return ret

	# retursive solution returns the list of all
	# permutations which start with the soFar list
	def getChoicesHelper(self, soFar):
		# base case
		if len(soFar) == self.getN():
			return [soFar]

		# get legal moves
		choices = set(range(0, self.getN()))
		chosen = set(soFar)
		left = choices - chosen

		# try each legal move
		ret = []
		for choice in left:
			soFarCopy = copy.deepcopy(soFar)
			soFarCopy += [choice]
			ret += self.getChoicesHelper(soFarCopy)
		return ret

class PermuteDict(Decision):
	def renderCode(self):
		return '\n'.join(self.params['dict'].values())
		# lines = []
		# for key in self.params['dict']:
		# 	lines.append(self.params['dict'][key])
		# return self.expand('PermuteLines', {
		# 	'id':self.params['id'],
		# 	'lines':lines
		# })