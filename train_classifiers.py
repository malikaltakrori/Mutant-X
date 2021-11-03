import sys, os
from joblib import dump, load
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

from absl import app, logging, flags

import concurrent.futures as futures

import writeprintsStatic

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC, SVC

sys.path.insert(1, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from Misc.utils import FLAGS

# # region Flags definition
# FLAGS = flags.FLAGS
#
# if "flags_defined" not in globals():
# 	# mode
# 	flags.DEFINE_enum('running_mode', None, ['debug', 'silent', 'info'], 'setting the verbosity level')
#
# 	# dataset
# 	flags.DEFINE_string('dataset_name', None, 'Dataset name')
# 	flags.DEFINE_integer("case", 1, "How to split the topics into: train/valid/test")
# 	flags.DEFINE_string('dataset_path', './myData', 'The path to my datasets folder')
#
# 	# Anonimizers
# 	flags.DEFINE_enum('anon', None, ['BT', 'ID', 'mutX', 'REAE'], 'Anonymization technique to use')
#
# 	flags.DEFINE_string("vocab_source", "same",
# 	                    "Whether to use an external dataset for vocabulary features ['same', <external_dataset_name>]")
#
# 	# flags.DEFINE_bool('small_dataset', False, 'if we use small sample of the dataset for authorship')
# 	# flags.DEFINE_enum('model_name', None, ['DomainAdv', 'onePiece'], 'The model name')
#
# 	flags.DEFINE_string('embed_path', './embedx', 'The path to pretrained word embeddings')
# 	flags.DEFINE_enum('embed_init', None, ['random', '1hot', 'pretrained', 'ngrams', 'stylo'],
# 	                  'What initialization for the embedding matrix: \'random\', \'1hot\', \'pretrained\', \'ngrams\'')
# 	flags.DEFINE_integer('embedding_size', 50, 'Dimension for the pretrained embeddings. Suggested: '
# 	                                           '50, 100, 200, 300')
# 	flags.DEFINE_bool('update_emb', False, 'Whether the embedding matrix is fine tuned')
# 	flags.DEFINE_bool('lower_case', False, 'Whether to keep sentence case or force lower case')
# 	flags.DEFINE_bool('remove_sw', False, 'Whether to remove stopwords or keep them')
# 	flags.DEFINE_integer('freq_threshold', 5, 'Word corpus frequency in order to keep it in text')
# 	# flags.DEFINE_bool('reverse_grad', False, 'Whether to reverse gradient for DANN model or not')
# 	flags.DEFINE_integer('vocab_size', 1000, 'number of words in the vocabulary set')
# 	flags.DEFINE_integer('max_seq_len', 0, 'Maximum words per document')
# 	# flags.DEFINE_float('lamb', 1.0, 'Weight of the topic loss for the DANN model')
#
# 	flags.DEFINE_integer("k", 0, "The number of words to get either from the training set or the external one")
# 	flags.DEFINE_string("ngram_level", "word", "$n$-gram level ['word', 'char'].")
# 	flags.DEFINE_integer("min_n", 1, "Min value of n in (n-gram). Default:1")
# 	flags.DEFINE_integer("max_n", 1, "Max value of n in (n-gram). Default:1")
# 	flags.DEFINE_integer("min_freq", 1, "Minimum frequency for a word to have an embedding.")
# 	flags.DEFINE_integer("run_times", 1, "The number of times to repeat an experiment -classification part, Default:1")
# 	flags.DEFINE_bool('mask_digits', False, "Whether to mask digits with #")
# 	flags.DEFINE_enum('mode', None, ['source', 'domain', 'dann'], 'What model to run')
# 	flags.DEFINE_enum('scenario', None, ['same', 'cross'], "Whether authorship is same-topic or cross-topic")
# 	flags.DEFINE_integer('batch_size', 1, 'Batch size', lower_bound=1, upper_bound=16)
# 	flags.DEFINE_integer('epochs', 2, 'The number of epochs', lower_bound=2)
#
# 	flags.DEFINE_bool('verbose', False, 'Show output or supress it')
# 	flags.DEFINE_integer('randSeed', 15, 'Random seed value')
#
# flags_defined = True
# # endregion


sys.path.insert(1, os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]))
from Misc.myDatasets import getDS, loadMyData


def feat_extract(ds, FLAGS, fname=None):
	lines2Write = []
	pathToDS = FLAGS.dataset_name + '_case' + str(
		FLAGS.case) if FLAGS.dataset_name == '4_Guardian_new' else FLAGS.dataset_name
	pathToDS = os.path.join('classifiers', pathToDS) + fname

	continue_from = 0
	train_x, train_y = [], []
	if os.path.exists(pathToDS + '.csv', ):
		with open(pathToDS + '.csv', 'r') as existingOutput:
			allLines = existingOutput.readlines()
			continue_from = len(allLines)
			for line in allLines:
				split_line = line.split(',')
				train_y.append(split_line.pop(0))
				train_x.append([float(x) for x in split_line])

	print(f"Continue from {continue_from}") if FLAGS.verbose else print("", end='')

	pool = futures.ProcessPoolExecutor()
	temp_list, temp_list_y = [], []
	for i, (inputText, y) in enumerate(zip(ds['x_'][continue_from:], ds['y_'][continue_from:])):
		temp_list.append(inputText)
		temp_list_y.append(y)

		if i % 50 == 0:
			print(i + continue_from) if FLAGS.verbose else print("", end='')
			feats_x = list(pool.map(writeprintsStatic.calculateFeatures, temp_list))
			train_y.extend(temp_list_y)
			train_x.extend(feats_x)

			lines2Write = []
			for x_sample, y in zip(feats_x, temp_list_y):
				lines2Write.append(f"{y},{','.join([str(x) for x in x_sample])}\n")
			# lines2Write = [f"{y},{','.join([str(x) for x in sample])}\n" for sample in feats_x]

			with open(pathToDS + '.csv', 'a') as outF:
				outF.writelines(lines2Write)
				# lines2Write = []

			temp_list, temp_list_y = [], []

	if len(temp_list) > 0:
		feats_x = list(pool.map(writeprintsStatic.calculateFeatures, temp_list))
		train_y.extend(temp_list_y)
		train_x.extend(feats_x)

		lines2Write = []
		for x_sample, y in zip(feats_x, temp_list_y):
			lines2Write.append(f"{y},{','.join([str(x) for x in x_sample])}\n")
		# lines2Write = [f"{y},{','.join([str(x) for x in sample])}\n" for sample in feats_x]

		with open(pathToDS + '.csv', 'a') as outF:
			outF.writelines(lines2Write)
			print(f'the last {len(lines2Write)} texts') if FLAGS.verbose else print("", end='')

	return train_x, train_y


def train_model():
	if FLAGS.running_mode == 'debug':
		logging.set_verbosity(logging.DEBUG)
	elif FLAGS.running_mode == 'info':
		logging.set_verbosity(logging.INFO)
	elif FLAGS.running_mode == 'silent':
		logging.set_verbosity(logging.FATAL)

	datasets = getDS(FLAGS)
	print("data read successfully") if FLAGS.verbose else print("", end="")


	if FLAGS.dataset_name == '4_Guardian_new':
		train_x, train_y = feat_extract({'x_': datasets['x_train1'],
		                                 'y_': datasets['y_train1']},
		                                FLAGS, fname='train')

		valid_x, valid_y = feat_extract({'x_': datasets['x_train2'],
		                                 'y_': datasets['y_train2']},
		                                FLAGS, fname='valid')

		tests_x, tests_y = feat_extract({'x_': datasets['x_valid'] + datasets['x_tests'],
		                                 'y_': datasets['y_valid'] + datasets['y_tests']},
		                                FLAGS, fname='tests')
	else:
		train_x, train_y = feat_extract({'x_': datasets['x_train1'],
		                                 'y_': datasets['y_train1']},
		                                FLAGS, fname='train')

		valid_x, valid_y = feat_extract({'x_': datasets['x_valid'],
		                                 'y_': datasets['y_valid']},
		                                FLAGS, fname='valid')

		tests_x, tests_y = feat_extract({'x_': datasets['x_tests'],
		                                 'y_': datasets['y_tests']},
		                                FLAGS, fname='tests')



	m = np.min(train_x)
	x = np.max(train_x)

	ss = StandardScaler()
	#
	train_x = np.float32(ss.fit_transform(train_x))
	# valid_x = np.float32(ss.transform(valid_x))
	tests_x = np.float32(ss.transform(tests_x))

	# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	# train_x = imp.fit_transform(train_x)
	# # valid_x = imp.transform(valid_x)
	# tests_x = imp.transform(tests_x)

	# selector = VarianceThreshold(threshold=0)
	#
	# train_x = np.float32(selector.fit_transform(train_x))
	# # valid_x = np.float32(selector.transform(valid_x))
	# tests_x = np.float32(selector.transform(tests_x))

	# train_ model
	clf = SVC(probability=True)
	# clf = RFC(50)
	clf.fit(train_x, train_y)
	preds = clf.predict(tests_x)

	print(FLAGS.dataset_name, '{:.2f}'.format(100* balanced_accuracy_score(preds, tests_y)))


	# save model
	pathToDS = ''
	if FLAGS.dataset_name == '4_Guardian_new':
		pathToDS = FLAGS.dataset_name + '_case' + str(FLAGS.case)
	elif FLAGS.dataset_name == 'C50':
		pathToDS = FLAGS.dataset_name + '_' + str(FLAGS.c_size)
	else:
		pathToDS = FLAGS.dataset_name

	pathToDS = os.path.join('classifiers', pathToDS)
	dump(clf, pathToDS + '.sav')

	# test a saved model
	new_model = load(pathToDS + '.sav')
	# print(balanced_accuracy_score(new_model.predict(tests_x), tests_y))

	return 0

def main(argv):
	for ds in ['EBG_small', 'C5','C10']: #, 'C50', 'EBG_full']:
		if ds in ['C5','C10']:
			FLAGS.dataset_name = 'C50'
			if ds.endswith('5'):
				FLAGS.c_size = 5
			elif ds.endswith('10'):
				FLAGS.c_size = 10
		else:
			FLAGS.dataset_name = ds
		train_model()

	return 0

if __name__ == '__main__':
	app.run(main)
	# app.run(train_model)
