
import os
import json
import multiprocessing
import time

t_all_start = time.time()

##### set params
params = {
"T": 60,
"V": 1000,
"alpha": "[0.1] * K",
"n_docs_per_t": 100,
"n_words_per_doc": 100,
"n_iter": 30,
"n_epoch":1,
"alpha0": 1.0,
"beta0": 1.0,
"b0": 0.005,
"b1": 1.1,
"b2": 1.1,
"epsilon":0.2
}

params_chg = []

for cp in [1,2,4]:
	for K in [10,20]:
		if cp == 1:
			for i,betai in enumerate([[0.1,1.5],[0.15,2.5]]):
				tau = [30]
				params_chg.append([cp,tau,K,i,betai])
		if cp == 2:
			for i,betai in enumerate([[0.1,0.2,1.5],[0.05,0.25,2.5]]):
				tau = [20,40]
				params_chg.append([cp,tau,K,i,betai])
		if cp == 4:
			for i,betai in enumerate([[0.01,0.025,0.5,0.1,1.5],[0.01,0.05,0.10,0.15,2.5]]):
				tau = [12,24,36,48]
				params_chg.append([cp,tau,K,i,betai])
print(params_chg)

for parami in params_chg:

	cp = parami[0]
	params['tau'] = parami[1]
	params['K'] = parami[2]
	i = parami[3]
	params['beta'] = parami[4]
	params['b0'] = parami[4][0] - 0.01
	params['b1'] = parami[4][-1] + 0.01
	params['b2'] = parami[4][-1] + 0.01
	params['epsilon'] = round((params['b1'] - params['b0']) / 10, 3)

	# create experiment_dir
	experiment_dir = 'experiments/cp{}_k{}_beta{}'.format(str(cp),str(params['K']),str(i))
	if not os.path.exists(experiment_dir):
		os.mkdir(experiment_dir)

	# write params.json
	filepath = 'experiments/cp{}_k{}_beta{}/params.json'.format(str(cp),str(params['K']),str(i))
	with open(filepath, 'w') as f:
		json.dump(params, f)

	for pre in map(str, range(50)):
		print('------------ File: {} ----------------'.format(experiment_dir))
		print('------------ Prefix: {} --------------'.format(str(pre)))

		t_start = time.time()
		print('-- Generating data. -- {}'.format(time.asctime(time.localtime(t_start))))
		os.system('nohup python gendata.py --experiment_dir {0} --prefix {1} \
			> {0}/{1}_logs.txt 2>&1'.format(experiment_dir, pre))
		t_end = time.time()
		print('-- Generating data done. -- {} -- using {} mins.'.format(
			time.asctime(time.localtime(t_end)), str(round(t_end-t_start)//60)))

		t_start = time.time()
		print('-- Running dphmm. -- {}'.format(time.asctime(time.localtime(t_start))))
		os.system('nohup python dphmm.py --experiment_dir {0} --prefix {1} \
			> {0}/{1}_logs.txt 2>&1'.format(experiment_dir, pre))
		t_end = time.time()
		print('-- Running dphmm done. -- {} -- using {} mins.'.format(
			time.asctime(time.localtime(t_end)), str(round(t_end-t_start)//60)))

t_all_end = time.time()

print('All simulation done. -- {} -- using {} mins.'.format(
	time.asctime(time.localtime(t_all_end)), str(round(t_all_end-t_all_start)//60)))

