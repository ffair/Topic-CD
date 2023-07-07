#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils
from utils import Params
from utils import input_docs
from utils import extractDocParams
from utils import saveVocabs
from utils import initS
from utils import initBeta
from utils import initZ
from utils import updateS
from utils import updateNikv
from utils import updateBeta
from utils import sampleZ
from utils import saveCp
from utils import compute_pikv
from utils import savePhi_Top100
import os
import time
import argparse
import multiprocessing
#import random


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='cell/simu',
                    help='Directory containing params.json, docs.txt')
parser.add_argument('--prefix', default='0',
                    help='prefix for filename of docs')

################ load params ################
args = parser.parse_args()
prefix = args.prefix
json_path = os.path.join(args.experiment_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

# hyperparams
n_epoch = params.n_epoch
n_iter = params.n_iter
alpha0 = params.alpha0
beta0 = params.beta0
T = params.T
K = params.K
b0 = params.b0
b1 = params.b1
b2 = params.b2
epsilon = params.epsilon
alpha = eval(params.alpha)

true_tau = params.tau
true_beta = params.beta

# io path
path_doc = os.path.join(args.experiment_dir, '0_docs.txt')
path_vocab = os.path.join(args.experiment_dir, '{}_vocab.json'.format(args.prefix))
path_output = os.path.join(args.experiment_dir, '{}_output.txt'.format(args.prefix))
path_phi = os.path.join(args.experiment_dir, "{}_phi.txt".format(args.prefix))

################ input docs ################
docs = input_docs(T, path_doc)                ## [ [[doc],...,[doc]], [[doc],...,[doc]],..., [[doc],...,[doc]] ]

# extract doc related params
D, N, V, vocab, vocab_dict = extractDocParams(docs)
revkv_vocab_dict = dict(zip(range(V), vocab))
# save vocabs
saveVocabs(vocab, V, path_vocab)

################ training ################


for epoch in range(n_epoch):
    print("===============initialize=============")

    S = initS(T, 2)
    print("-done init S:{}", S)
    # P = len(set(S))

    # init beta
    betaI = initBeta(S, b0, b1)
    print("-done init betaI:{}", betaI)

    # init z, ntdk, nikv, ntd, nik
    Ztdn, Vtdn, ntdk, ntd, ntkv, ntk = initZ(T, docs, D, N, K, V, vocab_dict)
    print("-done init Z")

    print("===============update==============")
    for iter in range(n_iter):
        # update s
        s_start = time.time()
        S = updateS(T, S, alpha0, beta0, betaI, V, K, ntkv)
        s_end = time.time()
        print("iter:{}, S:{}".format(iter, S))

        # update nikv, nik
        i_start = time.time()
        nikv , nik = updateNikv(S, T, D, N, Ztdn, Vtdn, K, V)
        i_end = time.time()

        # update beta
        b_start = time.time()
        betaI = updateBeta(S, betaI, b2, epsilon, nikv, V, K)
        b_end = time.time()
        print("iter:{}, beta:{}".format(iter, [(k, v) for k, v in betaI.items() if k in set(S)]))

        # update Z
        z_start = time.time()


        def mapf_updatez(zlist):
            t, zdn, vdn, ndk, nkv, nk = zlist
            for d in range(D[t]):
                for n in range(N[t][d]):
                    k = zdn[d][n]
                    v = vdn[d][n]
                    ndk[d][k] -= 1
                    nkv[k][v] -= 1
                    nk[k] -= 1

                    k_new = sampleZ(S, t, d, v, ndk, nikv, nik, K, V, betaI, alpha)
                    zdn[d][n] = k_new
                    ndk[d][k_new] += 1
                    nkv[k_new][v] += 1
                    nk[k_new] += 1
            return (zdn, vdn, ndk, nkv, nk)


        def updateZ(T, Ztdn, Vtdn, ntdk, ntkv, ntk):
            zlist = list(zip(range(T), Ztdn, Vtdn, ntdk, ntkv, ntk))
            pool = multiprocessing.Pool(processes=16)
            res = pool.map(mapf_updatez, zlist)
            Ztdn = [z for (z, v, n1, n2, n3) in res]
            Vtdn = [v for (z, v, n1, n2, n3) in res]
            ntdk = [n1 for (z, v, n1, n2, n3) in res]
            ntkv = [n2 for (z, v, n1, n2, n3) in res]
            ntk = [n3 for (z, v, n1, n2, n3) in res]
            pool.close()
            pool.join()
            return Ztdn, Vtdn, ntdk, ntkv, ntk


        Ztdn, Vtdn, ntdk, ntkv, ntk = updateZ(T, Ztdn, Vtdn, ntdk, ntkv, ntk)
        z_end = time.time()
        print("iter:{},Z000:{},Z111:{}".format(iter, Ztdn[0][0][0], Ztdn[1][1][1]))

        # print endure
        print("endure for updating S={}".format(s_end - s_start))
        print("endure for updating nikv={}".format(i_end - i_start))
        print("endure for updating beta={}".format(b_end - b_start))
        print("endure for updating Z={}\n".format(z_end - z_start))
        print("----done iter:{}".format(iter))

        # save cps and betas
        if iter == n_iter - 1:
            saveCp(path_output, iter, S, betaI)
            print("iter:{}, done saving cp".format(iter))
            pikv = compute_pikv(betaI, nikv, nik, K, V)
            savePhi_Top100(path_phi, betaI, pikv, revkv_vocab_dict, K, V)
            print("save phi successfully")

