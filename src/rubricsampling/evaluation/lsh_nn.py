import pickle
import argparse
from datasketch import MinHashLSHForest, MinHash
import javalang
import operator
import sys
import statistics
from nltk.util import ngrams
from tqdm import tqdm
import matplotlib.pyplot as plt
from base_nn import ProgramNN

import src.utils.paths as paths
import os

from src.utils.io_utils import save_pickle, load_pickle, save_json, load_json


SYNTH_NAME = '/anon_mapping_shard_0.pkl'

class LshNN(ProgramNN):
	CACHE_DIR = 'cache/'

	def __init__(self, sampledDataPath, num_perm=128, top_k=1, evict_cache=False):
		"""
		An agent class to find rubric sampled nearest neighbour of a given
		program by using a MinHash LSH forest.

		"""
		self.sampledDataPath = sampledDataPath
		self.num_perm = num_perm
		self.top_k = top_k
		self.evict_cache = evict_cache
		self.rawProgramData, self.sampledData = self.loadSyntheticData()
		self.create_lsh_forest()


	def create_lsh_forest(self):
		cache_file = os.path.join(self.CACHE_DIR, 'lsh_forest.pkl')
		if not self.evict_cache and os.path.isfile(cache_file):
			# load precomputed
			print('Loading cached forest')
			self.forest = load_pickle(cache_file)
		else:
			sampledSets = self.processData(self.sampledData)
			self.sampledMinHashes = self.createMinHashSet(sampledSets)

			self.forest = MinHashLSHForest(num_perm=self.num_perm)
			for prog_idx, minHash in enumerate(self.sampledMinHashes):
				self.forest.add(prog_idx, minHash)

			self.forest.index()

			os.makedirs(self.CACHE_DIR, exist_ok=True)
			save_pickle(self.forest, cache_file)

	def minHash(self, code_tokens):
		minHash = MinHash(num_perm=self.num_perm)
		for d in code_tokens: # TODO modify this for n-grams
			minHash.update("".join(d).encode('utf-8'))

		return minHash

	# create minHash objects for every dataset
	def createMinHashSet(self, dataset):
		minHashes = []
		for code in tqdm(dataset):
			minHashes.append(self.minHash(code))
		return minHashes

	def multi_dict_get(self, key, all_dicts):
		for dic in all_dicts:
			if key in dic:
				return dic[key]
		raise ValueError('Key not in any of the dictionaries')

	def loadSyntheticData(self):
		cache_file = os.path.join(self.CACHE_DIR, 'lsh_programs.pkl')
		if not self.evict_cache and os.path.isfile(cache_file):
			data = load_json(cache_file)
			prog_items = data['raw_programs']
			anon_progs = data['anon_programs']
		else:
			standard_path = self.sampledDataPath + '/standard/train' + SYNTH_NAME
			uniform_path = self.sampledDataPath + '/uniform/train' + SYNTH_NAME
			tempered_path = self.sampledDataPath + '/tempered/train' + SYNTH_NAME
			standardDict = pickle.load(open(standard_path, "rb" ))
			uniformDict = pickle.load(open(uniform_path, "rb" ))
			temperedDict =  pickle.load(open(tempered_path, "rb" ))

			all_dicts = [standardDict, uniformDict, temperedDict]

			# this step is not stable across different runs if caching forest
			# so this needs to be cached too
			prog_items = list(standardDict.keys() | uniformDict.keys() | temperedDict.keys())
			anon_progs = [self.multi_dict_get(prog, all_dicts) for prog in prog_items]
			data = dict(raw_programs=prog_items, anon_programs=anon_progs)

			os.makedirs(self.CACHE_DIR, exist_ok=True)
			save_json(data, cache_file)

			# if we dont load cache here, we should regenerate forest too
			self.evict_cache = True

		return prog_items, anon_progs



	def transformCode(self, program):
		splitCode = program.split()
		return splitCode
		#return ngrams(splitCode, 3)

	# tokenize every sentence and return a list of sentences
	def processData(self, dataset):
		processed = []
		for datum in dataset:
			transformedCode = self.transformCode(datum)
			processed.append(transformedCode)
		return processed

	def findNearestNeighbours(self, studentProgram, **kwargs):
		minHash = self.minHash(self.transformCode(studentProgram))
		result = self.forest.query(minHash, self.top_k)
		top_k_programs_anon = [self.sampledData[idx] for idx in result]
		top_k_programs = [self.rawProgramData[idx] for idx in result]
		#return top_k_programs, top_k_programs_anon
		return top_k_programs


