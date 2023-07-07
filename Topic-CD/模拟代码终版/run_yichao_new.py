import os
import json
import multiprocessing
import time
import numpy as np

from gensim import corpora
from gensim.models import LdaSeqModel
import pandas as pd
import time
import sys
import pickle
import os
from gensim.models.wrappers.dtmmodel import DtmModel


t_all_start = time.time()

##### set params
params = {
"T": 100,
"V": 1000,
#"alpha": "[0.1] * K",
"n_docs_per_t": 100,
"n_words_per_doc": 100,
"n_iter": 60,
"n_epoch":1,
"alpha0": 1.0,
"beta0": 1.0,
"b0": 0.005,
"b1": 1.1,
"b2": 1.1,
"epsilon":0.2
}

params_chg = []

for cp in [1]:
	for K in [10]:
		if cp == 1:
			for i,betai in enumerate([[0.1,1]]):
				tau = [30]
				params_chg.append([cp,tau,K,i,betai])
		if cp == 2:
			for i,betai in enumerate([[0.1,1,10]]):
				tau = [20,40]
				params_chg.append([cp,tau,K,i,betai])
		if cp == 3:
			for i,betai in enumerate([[0.1,1,10,50]]):
				tau= [25,50,75]
				params_chg.append([cp,tau,K,i,betai])
print(params_chg)

for parami in params_chg:

	cp = parami[0]
	params['tau'] = parami[1]
	params['K'] = parami[2]
	i = parami[3]
	params['alpha'] = list(np.random.uniform(0.05,0.15,K))
	params['beta'] = parami[4]
	params['b0'] = parami[4][0] - 0.01
	params['b1'] = parami[4][-1] + 0.01
	params['b2'] = parami[4][-1] + 0.01
	params['epsilon'] = round((params['b1'] - params['b0']) / 10, 3)

	# create experiment_dir
	experiment_dir = 'experiments/cp{}_k{}_beta{}_vec'.format(str(cp),str(params['K']),str(i))
	if not os.path.exists(experiment_dir):
		os.makedirs(experiment_dir)

	# write params.json
	filepath = 'experiments/cp{}_k{}_beta{}_vec/params.json'.format(str(cp),str(params['K']),str(i))
	with open(filepath, 'w') as f:
		json.dump(params, f)

	for pre in map(str, range(10)):
		print('------------ File: {} ----------------'.format(experiment_dir))
		print('------------ Prefix: {} --------------'.format(str(pre)))

		t_start = time.time()
		print('-- Generating data. -- {}'.format(time.asctime(time.localtime(t_start))))
		os.system('python gendata.py --experiment_dir {0} --prefix {1}'.format(experiment_dir, pre))
		t_end = time.time()
		print('-- Generating data done. -- {} -- using {} mins.'.format(
			time.asctime(time.localtime(t_end)), str(round(t_end-t_start)//60)))

		t_start = time.time()
		print('-- Running dphmm. -- {}'.format(time.asctime(time.localtime(t_start))))
		os.system('python dphmm.py --experiment_dir {0} --prefix {1}'.format(experiment_dir, pre))
		t_end = time.time()
		print('-- Running dphmm done. -- {} -- using {} mins.'.format(
			time.asctime(time.localtime(t_end)), str(round(t_end-t_start)//60)))

	for iter in range(10):
		num_topics = K
		input_path = os.path.join(experiment_dir,'{}_docs.txt'.format(iter))
		output_path = os.path.join(experiment_dir,'{}_dtm.pkl'.format(iter))

		with open(input_path, 'r') as f:
			docs = f.readlines()

		nums = [int(doc.split(',')[0]) for doc in docs]
		time_slice = list(pd.Series(nums).value_counts().sort_index())
		texts = [doc.strip().split(',')[1].split() for doc in docs]
		dictionary = corpora.Dictionary(texts)
		corpus = [dictionary.doc2bow(text) for text in texts]

		print('Start to run cp{}_k{}_beta{}_{} DTM.'.format(str(cp),str(K),str(i),str(iter)))
		print(time.asctime(time.localtime()))

		dtm_path = "./dtm-linux64"
		dtm_model = DtmModel(dtm_path, corpus, time_slice, num_topics=num_topics, id2word=dictionary)
		print(time.asctime(time.localtime()))

		print('Start to save model.')
		with open(output_path,'wb') as f:
			pickle.dump(dtm_model,f)
		print('DTM model has been saved to {}.'.format(output_path))
		print(time.asctime(time.localtime()))

t_all_end = time.time()

print('All simulation done. -- {} -- using {} mins.'.format(
	time.asctime(time.localtime(t_all_end)), str(round(t_all_end-t_all_start)//60)))

