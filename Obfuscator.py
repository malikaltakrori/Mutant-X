import argparse
import random
import math
import gensim.models.keyedvectors as word2vec
from Document import Document
from Classifier import Classifier
from Mutant import Mutant_X
from operator import itemgetter
import copy
import os, sys
import re
import numpy as np
import time
import io
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import writeprintsStatic
from shutil import copyfile
from pathlib import Path
import concurrent.futures

def computeNeighbours(word, wordVectorModel, neighborsCount=5):
	word = str(word).lower()
	tim = 0
	try:
		start = time.time()
		neighbors = list(wordVectorModel.similar_by_word(word, topn=neighborsCount))
		end= time.time()
		tim = end - start
	except:
		return -1,tim

	updated_neigbours = []
	for neighbor in neighbors:
		if neighbor[1] > 0.75:
			updated_neigbours.append(neighbor[0])
	if not updated_neigbours:
		return -1,tim
	# word_selected = random.choice(updated_neigbours)

	return updated_neigbours,tim

def getWordNeighboursDictionary(inputText,totalNeighbours):
	wordVectorModel = word2vec.KeyedVectors.load_word2vec_format("Word Embeddings/Word2vec.bin",
																 binary=True)
	average_time = []
	word_neighbours = {}
	for word in inputText:

		neighbors,tim = computeNeighbours(word, wordVectorModel,totalNeighbours)
		if len(average_time)!=100 and tim!=0:
			average_time.append(tim)

		if neighbors != -1:
			word_neighbours[word] = neighbors
	return word_neighbours

def getRunAndDocumentNumber(indexNo, dataset_name, authorsToKeep):
	indexNo = int(indexNo)
	indexNo = indexNo - 1

	if dataset_name == 'amt' and authorsToKeep == 5:
		indexNo = indexNo % 300
	elif dataset_name == 'amt' and authorsToKeep == 10:
		indexNo = indexNo % 490
	elif dataset_name == 'BlogsAll' and authorsToKeep == 5:
		indexNo = indexNo % 1000
	elif dataset_name == 'BlogsAll' and authorsToKeep == 10:
		indexNo = indexNo % 2000

	passNo = indexNo % 10
	passNo = passNo + 1
	documentNumber = math.floor(indexNo / 10)
	documentNumber = documentNumber + 1
	random.seed()

	return passNo, documentNumber

def saveDocument(document,qualitative_results_folder, iteration_m, obfuscated=False):
	if obfuscated:
		qualitativeoutputResults = open(qualitative_results_folder + "Obfuscated_text", "w")
	qualitativeoutputResults.write(document.documentText)

def getInformationOfInputDocument(documentPath):
	authorslabeler = LabelEncoder()
	authorslabeler.classes_ = np.load('classes.npy')

	inputText = io.open(documentPath, "r", errors="ignore").readlines()
	inputText = ''.join(str(e) + "" for e in inputText)

	authorName = (documentPath.split('/')[-1]).split('_')[0]
	authorLabel = authorslabeler.transform([authorName])[0]

	return (authorLabel, authorName, inputText)

def main():

	#################################################
	# Parameters
	#################################################
	parser = argparse.ArgumentParser()

	# Mutant_X parameters
	parser.add_argument("--generation", "-l", help="Number of documents generated per document", default=5)
	parser.add_argument("--topK", "-k", help="Top K highest fitness selection", default=5)
	parser.add_argument("--crossover", "-c", help="Crossover probability", default=0.1)
	parser.add_argument("--iterations", "-M", help="Total number of iterations", default=25)
	parser.add_argument("--alpha", "-a", help="weight assigned to probability in fitness function", default=0.75)
	parser.add_argument("--beta", "-b", help="weight assigned to METEOR in fitness function", default=0.25)
	parser.add_argument("--replacements", "-Z", help="percentage of document to change", default=0.05)
	parser.add_argument("--replacementsLimit", "-rl", help="replacements limit", default=0.20)

	# Obfuscator parameters
	parser.add_argument("--authorstoKeep", "-atk", help="Total number of authors under observation", default=5)
	parser.add_argument("--datasetName", "-dn", help="Name of dataset to test with", default='amt')
	parser.add_argument("--allowedNeighbours", "-an", help="Total allowed neighbours in word embedding", default=5)
	parser.add_argument("--documentName", "-docN", help="Name of document for obfuscation", default='h_13_2.txt')
	parser.add_argument("--classifierType", "-ctype", help="Type of classifier", default='ml')


	args = parser.parse_args()

	generation = int(args.generation)
	topK = int(args.topK)
	crossover = float(args.crossover)
	iterations = int(args.iterations)
	alpha = float(args.alpha)
	beta = float(args.beta)
	replacements = float(args.replacements)
	replacementsLimit = float(args.replacementsLimit)

	authorstoKeep = int(args.authorstoKeep)
	datasetName = args.datasetName
	documentName = args.documentName
	allowedNeighbours = int(args.allowedNeighbours)
	classifierType = args.classifierType

	runNumber = 1

	####################################################
	# Loading Document to be Obfuscated
	####################################################

	clf = Classifier(classifierType, authorstoKeep, datasetName, documentName)
	clf.loadClassifier()
	# testInstancesFilename = "../../Data/datasetPickles/" + str(datasetName) + '-' + str(authorstoKeep) + '/X_test.pickle'
	# with open(testInstancesFilename, 'rb') as f:
	#     testInstances = pickle.load(f)
	#
	# print("Test Instances Length : ", len(testInstances))
	filePath, filename = documentName, documentName
	authorId, author, inputText = getInformationOfInputDocument(filePath)

	print("Document Name : ", filename)

	originalDocument = Document(inputText)
	clf.getLabelAndProbabilities(originalDocument)

	if originalDocument.documentAuthor != authorId:
		print("Classified InCorrectly , Hence Skipping the Article")
		return

	if not os.path.exists('documentsWordsSpace'):
		os.makedirs('documentsWordsSpace')

	if os.path.isfile('documentsWordsSpace/' + filename.split('.')[0] + '.pickle'):
		print('Word Space Dictionary exists !!')
		with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'rb') as handle:
			neighboursDictionary = pickle.load(handle)
	else:
		print('Creating Word Space Dictionary !!')
		neighboursDictionary = getWordNeighboursDictionary(originalDocument.documentWords,allowedNeighbours)
		with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'wb') as handle:
			pickle.dump(neighboursDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


	####################################################
	# Setting up directories and loading Mutant_X
	####################################################

	mutant_x = Mutant_X(generation, topK, crossover, iterations, neighboursDictionary, alpha, beta, replacements, replacementsLimit)

	print('ARTICLE ====> ', filename, "    Author ====> ", author, "     Pass NO. ====> ", runNumber)

	runNumber = str(runNumber) + "_" + str(time.time())

	prob_str = ''
	prob_values = ''
	for i in range(authorstoKeep):
		prob_str += ','
		prob_str += str(i)
		prob_values+= ','
		prob_values+=str(originalDocument.documentAuthorProbabilites[i])

	####################################################
	# Starting Mutant_X Obfuscation Process
	####################################################

	indivisualsPopulation = [originalDocument]
	iteration_m = 1
	obfuscated = False
	while (iteration_m < mutant_x.iterations) and (obfuscated == False) :

		print("Iteration =====> ", iteration_m)

		# Generation Process
		generatedPopulation = []
		for indivisual in indivisualsPopulation:
			for i in range(0, mutant_x.generation):
				indivisualCopy = copy.deepcopy(indivisual)
				genDocument = mutant_x.makeReplacement(indivisualCopy)
				clf.getLabelAndProbabilities(genDocument)
				generatedPopulation.append(genDocument)

		indivisualsPopulation.extend(generatedPopulation)

		# Crossover Process
		if random.random() < mutant_x.crossover:
			print("CROSSING OVER")
			choice1, choice2 = random.sample(indivisualsPopulation, 2)
			choice1Copy = copy.deepcopy(choice1)
			choice2Copy = copy.deepcopy(choice2)
			child_1, child_2 = mutant_x.single_point_crossover(choice1Copy, choice2Copy)

			clf.getLabelAndProbabilities(child_1)
			clf.getLabelAndProbabilities(child_2)

			indivisualsPopulation.extend([child_1, child_2])


		if originalDocument in indivisualsPopulation:
			indivisualsPopulation.remove(originalDocument)

		# Calculating Fitness

		for indivisual in indivisualsPopulation:

			mutant_x.calculateFitness(originalDocument,indivisual)

			if indivisual.documentAuthor != originalDocument.documentAuthor:
				print("Obfuscated Successfully !!!")
				obfuscated = True
				saveDocument(indivisual, '', 0,obfuscated=True)


			prob_str = ''
			for i in range(authorstoKeep):
				prob_str += ','
				prob_str += str(indivisual.documentAuthorProbabilites[i])


		# Selecting topK

		indivisualsPopulation.sort(key=lambda x:x.fitnessScore, reverse=True)
		indivisualsPopulation = indivisualsPopulation[:mutant_x.topK]

		iteration_m+=1


def mutx_anonymizeIMDB62(FLAGS):
	if os.path.isdir(os.path.join(FLAGS.dataset_path, 'anonymized')):
		if os.path.isdir(os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}')):
			pass
		else:
			os.mkdir(os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}'))
	else:
		print('As a fail-safe, create the {} directory manually'.format(os.path.join(FLAGS.dataset_path, 'anonymized')))

	new_path = os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}')
	# create dummy files for train and valid

	# copy train (and valid) as it is
	full_path = os.path.join(FLAGS.dataset_path, FLAGS.dataset_name)
	src = os.path.join(full_path, FLAGS.dataset_name )
	target = os.path.join(new_path, FLAGS.dataset_name )

	copyfile(src + '_train.tsv', target + '_train.tsv')
	copyfile(src + '_valid.tsv', target + '_valid.tsv')

	with open(os.path.join(full_path, 'IMDB62_test.tsv'), 'r') as inF:#, open(os.path.join(new_path, 'IMDB62_test.tsv'), 'w') as outF:
		alllines = inF.readlines()

		continue_from = 0
		if os.path.exists(os.path.join(new_path, 'IMDB62_test.tsv')):
			with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'r') as existingOutput:
				continue_from = len(existingOutput.readlines())
		else:
			# create file if dne
			with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'w') as existingOutput:
				pass

		temp_lines = []
		temp_ids = []
		lines2Write = []
		for i, l in enumerate(alllines[continue_from:]):
			# p = Posts()
			l = l.split('\t')
			cont = l[0]
			p_author_id = int(l[1])

			# lines2Write.append(f"{cont}\t{p_author_id}\n")
			temp_lines.append(cont)
			temp_ids.append(p_author_id)

			if i % 50 == 0:
				t1 = time.time()
				# with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'a') as outF:
				# 	outF.writelines(lines2Write)
				# lines2Write = []

				# pool = concurrent.futures.ProcessPoolExecutor()
				# anonymized_docs = list(pool.map(anonymizer, temp_lines))
				with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
					futures = [executor.submit(anonymizer, doc) for doc in temp_lines]
					anonymized_docs = [f.result() for f in futures]

				for p, id in zip(anonymized_docs, temp_ids):
					lines2Write.append(f"{p}\t{id}\n")

				with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'a') as outF:
					outF.writelines(lines2Write)
					temp_lines = []
					lines2Write = []

				# time.sleep(2)
				print(i+continue_from,end='\t')
				print(time.time()-t1)
				# takes 7 seconds


				# t1 = time.time()
				# p_content = anonymizer(temp_lines)
				# for p, id in zip(p_content, temp_ids):
				# 	lines.append(f"{p}\t{id}\n")
				#
				# with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'w+') as outF:
				# 	outF.writelines(lines)
				# temp_lines = []
				# temp_ids = []
				# print(time.time()-t1)
				# takes 76 seconds

		# if len(lines2Write) > 0:
		# 	with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'a') as outF:
		# 		outF.writelines(lines2Write)
	#
		lines2Write = []
		print(len(temp_lines))
		with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
			futures = [executor.submit(anonymizer, doc) for doc in temp_lines]
			anonymized_docs = [f.result() for f in futures]
			for p, id in zip(anonymized_docs, temp_ids):
				lines2Write.append(f"{p}\t{id}\n")

		with open(os.path.join(new_path, 'IMDB62_test.tsv'), 'a') as outF:
			outF.writelines(lines2Write)

	return

def mutx_anonymizeGuardian(FLAGS):
	full_path = os.path.join(FLAGS.dataset_path, FLAGS.dataset_name)

	if os.path.isdir(os.path.join(FLAGS.dataset_path, 'anonymized')):
		if os.path.isdir(os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}')):
			pass
		else:
			os.mkdir(os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}'))
	else:
		print('As a fail-safe, create the {} directory manually'.format(os.path.join(FLAGS.dataset_path, 'anonymized')))

	new_path = os.path.join(FLAGS.dataset_path, 'anonymized', f'{FLAGS.dataset_name}_{FLAGS.anon}')

	for topic in sorted(os.listdir(full_path)):
		path = os.path.join(full_path, topic)
		if os.path.isdir(path):
			authorstoKeep = len(os.listdir(path))
			for author in sorted(os.listdir(path)):
				subpath = os.path.join(path, author)
				Path(os.path.join(new_path, topic, author)).mkdir(parents=True, exist_ok=True)
				for filename in sorted(os.listdir(subpath)):
					# this should allow resuming for the guardian
					if os.path.exists(os.path.join(new_path, topic,author,filename)):
						continue

					fpath = os.path.join(subpath, filename)
					# if sys.version_info < (3,):
					# 	f = open(fpath)
					# else:
					# 	f = open(fpath, encoding='latin-1')
					# t = f.read()
					anon_t = use_Externally(authorstoKeep=authorstoKeep, datasetName=FLAGS.dataset_name, documentName= fpath)
					with open(os.path.join(new_path, topic,author,filename), 'w') as anonFile:
						anonFile.writelines(anon_t)

					# f.close()


def mutx_anonymizeNewYelp(FLAGS):
	raise NotImplementedError

def use_Externally(
		authorstoKeep,
		datasetName,
		documentName):

	#################################################
	# Parameters
	#################################################
	# parser = argparse.ArgumentParser()
	#
	# # Mutant_X parameters
	# parser.add_argument("--generation", "-l", help="Number of documents generated per document", default=5)
	# parser.add_argument("--topK", "-k", help="Top K highest fitness selection", default=5)
	# parser.add_argument("--crossover", "-c", help="Crossover probability", default=0.1)
	# parser.add_argument("--iterations", "-M", help="Total number of iterations", default=25)
	# parser.add_argument("--alpha", "-a", help="weight assigned to probability in fitness function", default=0.75)
	# parser.add_argument("--beta", "-b", help="weight assigned to METEOR in fitness function", default=0.25)
	# parser.add_argument("--replacements", "-Z", help="percentage of document to change", default=0.05)
	# parser.add_argument("--replacementsLimit", "-rl", help="replacements limit", default=0.20)
	#
	# # Obfuscator parameters
	# parser.add_argument("--authorstoKeep", "-atk", help="Total number of authors under observation", default=5)
	# parser.add_argument("--datasetName", "-dn", help="Name of dataset to test with", default='amt')
	# parser.add_argument("--allowedNeighbours", "-an", help="Total allowed neighbours in word embedding", default=5)
	# parser.add_argument("--documentName", "-docN", help="Name of document for obfuscation", default='h_13_2.txt')
	# parser.add_argument("--classifierType", "-ctype", help="Type of classifier",default='ml')
	#
	#
	# args = parser.parse_args()

	generation = 5
	topK = 5
	crossover = 0.1
	iterations = 25
	alpha = 0.75
	beta = 0.25
	replacements = 0.05
	replacementsLimit = 0.20

	# authorstoKeep = int(args.authorstoKeep)
	# datasetName = args.datasetName
	# documentName = args.documentName
	allowedNeighbours = 5
	classifierType = 'ml'

	# generation = int(args.generation)
	# topK = int(args.topK)
	# crossover = float(args.crossover)
	# iterations = int(args.iterations)
	# alpha = float(args.alpha)
	# beta = float(args.beta)
	# replacements = float(args.replacements)
	# replacementsLimit = float(args.replacementsLimit)
	#
	# # authorstoKeep = int(args.authorstoKeep)
	# # datasetName = args.datasetName
	# # documentName = args.documentName
	# allowedNeighbours = int(args.allowedNeighbours)
	# classifierType = args.classifierType

	runNumber = 1

	####################################################
	# Loading Document to be Obfuscated
	####################################################

	clf = Classifier(classifierType, authorstoKeep, datasetName, documentName)
	clf.loadClassifier()
	# testInstancesFilename = "../../Data/datasetPickles/" + str(datasetName) + '-' + str(authorstoKeep) + '/X_test.pickle'
	# with open(testInstancesFilename, 'rb') as f:
	#     testInstances = pickle.load(f)
	#
	# print("Test Instances Length : ", len(testInstances))
	filePath, filename = documentName, documentName
	authorId, author, inputText = getInformationOfInputDocument(filePath)

	print("Document Name : ", filename)

	originalDocument = Document(inputText)
	clf.getLabelAndProbabilities(originalDocument)

	if originalDocument.documentAuthor != authorId:
		print("Classified InCorrectly , Hence Skipping the Article")
		return

	if not os.path.exists('documentsWordsSpace'):
		os.makedirs('documentsWordsSpace')

	if os.path.isfile('documentsWordsSpace/' + filename.split('.')[0] + '.pickle'):
		print('Word Space Dictionary exists !!')
		with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'rb') as handle:
			neighboursDictionary = pickle.load(handle)
	else:
		print('Creating Word Space Dictionary !!')
		neighboursDictionary = getWordNeighboursDictionary(originalDocument.documentWords, allowedNeighbours)
		with open('documentsWordsSpace/' + filename.split('.')[0] + '.pickle', 'wb') as handle:
			pickle.dump(neighboursDictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


	####################################################
	# Setting up directories and loading Mutant_X
	####################################################

	mutant_x = Mutant_X(generation, topK, crossover, iterations, neighboursDictionary, alpha, beta, replacements, replacementsLimit)

	print('ARTICLE ====> ', filename, "    Author ====> ", author, "     Pass NO. ====> ", runNumber)

	runNumber = str(runNumber) + "_" + str(time.time())

	prob_str = ''
	prob_values = ''
	for i in range(authorstoKeep):
		prob_str += ','
		prob_str += str(i)
		prob_values+= ','
		prob_values+=str(originalDocument.documentAuthorProbabilites[i])

	####################################################
	# Starting Mutant_X Obfuscation Process
	####################################################

	indivisualsPopulation = [originalDocument]
	iteration_m = 1
	obfuscated = False
	while (iteration_m < mutant_x.iterations) and (obfuscated == False) :

		print("Iteration =====> ", iteration_m)

		# Generation Process
		generatedPopulation = []
		for indivisual in indivisualsPopulation:
			for i in range(0, mutant_x.generation):
				indivisualCopy = copy.deepcopy(indivisual)
				genDocument = mutant_x.makeReplacement(indivisualCopy)
				clf.getLabelAndProbabilities(genDocument)
				generatedPopulation.append(genDocument)

		indivisualsPopulation.extend(generatedPopulation)

		# Crossover Process
		if random.random() < mutant_x.crossover:
			print("CROSSING OVER")
			choice1, choice2 = random.sample(indivisualsPopulation, 2)
			choice1Copy = copy.deepcopy(choice1)
			choice2Copy = copy.deepcopy(choice2)
			child_1, child_2 = mutant_x.single_point_crossover(choice1Copy, choice2Copy)

			clf.getLabelAndProbabilities(child_1)
			clf.getLabelAndProbabilities(child_2)

			indivisualsPopulation.extend([child_1, child_2])


		if originalDocument in indivisualsPopulation:
			indivisualsPopulation.remove(originalDocument)

		# Calculating Fitness

		for indivisual in indivisualsPopulation:

			mutant_x.calculateFitness(originalDocument,indivisual)

			if indivisual.documentAuthor != originalDocument.documentAuthor:
				print("Obfuscated Successfully !!!")
				obfuscated = True
				saveDocument(indivisual, '', 0,obfuscated=True)


			prob_str = ''
			for i in range(authorstoKeep):
				prob_str += ','
				prob_str += str(indivisual.documentAuthorProbabilites[i])


		# Selecting topK

		indivisualsPopulation.sort(key=lambda x:x.fitnessScore, reverse=True)
		indivisualsPopulation = indivisualsPopulation[:mutant_x.topK]

		iteration_m+=1


if __name__ == "__main__":
	main()