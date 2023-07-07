import numpy as np
from utils import Params
from utils import save_dict_to_json
import os
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/single_20',
                    help='Directory containing params.json')
parser.add_argument('--prefix', default='0',
                    help='prefix for filename of docs')

############## load params ##############
args = parser.parse_args()
json_path = os.path.join(args.experiment_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

# T, tau, P, S
T = params.T
tau = params.tau
tau.append(T)
P = len(tau) + 1
S = [0] * tau[0]
for i_cp in range(1, len(tau)):
    S += [i_cp] * (tau[i_cp] - tau[i_cp - 1])

# K, V, n_docs_per_t, n_words_per_doc, D, N
K = params.K
V = params.V
n_docs_per_t = params.n_docs_per_t
n_words_per_doc = params.n_words_per_doc

D = [max(n_docs_per_t, Dt) for Dt in np.random.poisson(n_docs_per_t, T)]  # n_docs
N = [np.random.poisson(n_words_per_doc, Dt) + 50 for Dt in D]  # n_words within a doc

# alpha, beta, phi
#alpha = eval(params.alpha)
alpha = params.alpha
beta0 = params.beta
beta = []
phi = []
for i in beta0:
    betai = [i] * V
    beta.append(betai)  # symmetric dirichlet
    phi.append(np.random.dirichlet(betai, K))

# save true phi
path_true_phi = os.path.join(args.experiment_dir, '{}_true_phi.json'.format(args.prefix))
phi_list = [p.tolist() for p in phi]
dic = dict(zip(beta0, phi_list))
save_dict_to_json(dic, path_true_phi)

############## generate & save data ##############
# generate docs
def sampleOneWord(s_theta):
    s = s_theta[0]
    theta = s_theta[1]
    z_vec = np.random.multinomial(1, theta)
    z = int(np.where(z_vec == 1)[0])
    w_vec = np.random.multinomial(1, phi[s][z])
    w = int(np.where(w_vec == 1)[0])
    return w

def wordsNumToWords(s_and_n):
    s = s_and_n[0]
    n = s_and_n[1]
    theta = np.random.dirichlet(alpha)
    doci = list(map(sampleOneWord, zip([s]*n,[theta]*n)))
    return doci

def momentToWords(moment):
    print('moment {} start dong.'.format(str(moment[0])))
    s = moment[1]
    N_t = moment[2]
    momenti = list(map(wordsNumToWords,zip([s]*len(N_t),N_t)))
    print('moment {} done.'.format(str(moment[0])))
    return momenti

#pool = multiprocessing.Pool(processes=8)
#words_vec = pool.map(momentToWords, zip(range(T),S,N))
words_vec = list(map(momentToWords, zip(range(T),S,N)))

# save docs
doc_path = os.path.join(args.experiment_dir, '{}_docs.txt'.format(args.prefix))
with open(doc_path, 'w') as f:
    for t in range(T):
        for d in words_vec[t]:
            f.write(str(t) + ',' + ' '.join(map(str, d)) + '\n')
        print("done t={}".format(t))
