import pickle
import argparse
from datasketch import MinHashLSHForest, MinHash
import javalang
import operator
import sys
import statistics
from nltk import ngrams
from tqdm import tqdm
import matplotlib.pyplot as plt
from base_nn import ProgramNN

import src.utils.paths as paths 

# This script helps you visualize the relative similarity of 
# your samples to student code using LSH.
# see: https://ekzhu.github.io/datasketch/lshforest.html
# To run:
# (1) change the path names above as needed 
# (2) modify the calls to computeLshNN if you have fewer sampled datasets
# (3) modify constructHistogram accordingly!

# load the synthetic and student data
# merge synthetic into one big (list?)
# compute LSH using that list
# print out the closest neighbor for a student program

# make it a class with init and a method that takes in student program and outputs nearest neighbor
# change it so that you can pass in arguments from command line

STUDENT_NAME = '/liftoff_train_anon.pkl'
SYNTH_NAME = '/anon_mapping_shard_0.pkl'

class LshSamplesEval:

	'''STUDENT_PATH = '../studentData/liftoff/'
	STANDARD_PATH = '../data/raw/liftoff/standard/'
	UNIFORM_PATH = '../data/raw/liftoff/uniform/'
	TEMPERED_PATH = '../data/raw/liftoff/tempered/'''

	def __init__(self, studentDataPath, sampledDataPath):
		print('Loading data...')
		self.studentDataPath = studentDataPath
		print(self.studentDataPath)
		self.sampledDataPath = sampledDataPath
		print(self.sampledDataPath)
		self.sampledData = self.loadSyntheticData()
		self.studentData = self.loadStudentData()

	def loadStudentData(self):
		path = self.studentDataPath + STUDENT_NAME
		datadict = pickle.load(open(path, "rb" ))
		return list(datadict.keys())

	def loadSyntheticData(self):
		standard_path = self.sampledDataPath + '/standard/train' + SYNTH_NAME
		uniform_path = self.sampledDataPath + '/uniform/train' + SYNTH_NAME
		tempered_path = self.sampledDataPath + '/tempered/train' + SYNTH_NAME
		standardDict = pickle.load(open(standard_path, "rb" ))
		uniformDict = pickle.load(open(uniform_path, "rb" ))
		temperedDict =  pickle.load(open(tempered_path, "rb" ))
		#import pdb; pdb.set_trace()
		return list(standardDict.values()) + list(uniformDict.values()) + list(temperedDict.values())

	def computeLshNN(self):
		print('Processing sampled and student data...')
		sampledSets = self.processData(self.sampledData)
		studentSets = self.processData(self.studentData)

		print('Finding nearest neighbors from sampled data...')
		sampledScores = self.constructNNList(studentSets, sampledSets, self.studentData, self.sampledData)

		print('Found nearest neighbors for data!')

		# self.constructHistogram(sampledScores)

		return sampledScores

	# tokenize every sentence and return a list of sentences
	def processData(self, dataset):
		processed = []
		for datum in dataset:
			splitCode = datum.split()
			processed.append(splitCode)
		return processed

	# runs MinHashLsh
	def constructNNList(self, studentSets, sampledSets, studentData, sampledData):
		print('Creating min-hashes for student data')
		self.studentMinHashes = self.createMinHash(studentSets)
		print('Creating min-hashes for rubric data')
		self.sampledMinHashes = self.createMinHash(sampledSets)

		self.forest = MinHashLSHForest(num_perm = 128)
		i = 0
		for minHash in self.sampledMinHashes:
			self.forest.add(str(i), minHash)
			i += 1

		self.forest.index()

		print("calculating nearest neighbor")
		scores = []
		for i, query in enumerate(tqdm(self.studentMinHashes)):
			result = self.forest.query(query, 1)
			indexMatch = int(result[0])
			# Uncomment these to print examples of 
			# student code and their nearest neighbor!
			print(result)
			print('Student Code: \n')
			print(studentData[i])
			print('\n')
			print('Closest Sampled Code: \n')
			print(sampledData[indexMatch])
			print('\n')
			score = self.sampledMinHashes[indexMatch].jaccard(query)
			print('Score: \n')

			scores.append(score)

		return scores

	# create minHash objects for every dataset
	def createMinHash(self, dataset):
		minHashes = []
		for code in tqdm(dataset):
			minHash = MinHash(num_perm = 128)
			for d in code: # TODO modify this for n-grams
				minHash.update("".join(d).encode('utf-8'))
			minHashes.append(minHash)
		return minHashes

	def constructHistogram(self, scores):
		plt.hist(scores)
		plt.xlabel('Jaccard Similarity Score')
		plt.ylabel('Counts')
		plt.show()


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		'studentDataDir',
		metavar='student-dir',
		default='None',
		help='The path to the student anonymized data')
	arg_parser.add_argument(
		'sampledDataDir',
		metavar='sampled-dir',
		default='None',
		help='The path to the anonymized sampled data path (with folders for each sampling style within it')

	args = arg_parser.parse_args()
	lsh = LshSamplesEval(args.studentDataDir, args.sampledDataDir)
	lsh.computeLshNN()

if __name__ == '__main__':
	main()



