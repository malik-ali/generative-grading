import sys
from tqdm import tqdm
from base_nn import ProgramNN

import numpy as np
import src.utils.paths as paths 
import os
import pickle

from src.utils.io_utils import save_pickle, load_pickle, save_json, load_json

SYNTH_NAME = '/anon_mapping_shard_0.pkl'

class RandomNN(ProgramNN):
	'''
	Baseline that returns a random program as the NN
	'''

	def __init__(self, sampledDataPath, top_k=1):
		"""
		An agent class to find rubric sampled nearest neighbour of a given
		program by randomly picking a NN

		"""
		self.sampledDataPath = sampledDataPath
		self.top_k = top_k
		self.rawProgramData = self.loadSyntheticData()
		
	def loadSyntheticData(self):
		standard_path = self.sampledDataPath + '/standard/train' + SYNTH_NAME
		uniform_path = self.sampledDataPath + '/uniform/train' + SYNTH_NAME
		tempered_path = self.sampledDataPath + '/tempered/train' + SYNTH_NAME
		standardDict = pickle.load(open(standard_path, "rb" ))
		uniformDict = pickle.load(open(uniform_path, "rb" ))
		temperedDict =  pickle.load(open(tempered_path, "rb" ))

		return list(standardDict.keys() | uniformDict.keys() | temperedDict.keys())

	def findNearestNeighbours(self, studentProgram, **kwargs):
		result = np.random.choice(len(self.rawProgramData), size=self.top_k, replace=False)
		top_k_programs = [self.rawProgramData[idx] for idx in result]
		return top_k_programs 
