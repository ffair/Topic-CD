#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import numpy as np
import math
from multiprocessing.dummy import Pool as Threadpool
from collections import Counter
import warnings

warnings.filterwarnings("error")


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


######## utils for inputing docs ########
def input_docs(T, path_doc):
    docs = [[] for t in range(T)]
    with open(path_doc, 'r') as f:
        for l in f:
            t, doc = l.strip().split(',')
            t = int(t)
            doc = doc.split(' ')
            docs[t].append(doc)
    return docs


def extractDocParams(docs):
    D = [len(dt) for dt in docs]
    N = [[len(doc) for doc in dt] for dt in docs]
    words = [w for dt in docs for doc in dt for w in doc]
    cnt = Counter(words)
    vocab = list(cnt.keys())
    V = len(np.array(vocab))
    vocab_dict = dict(zip(vocab, range(V)))
    return D, N, V, vocab, vocab_dict


def saveVocabs(vocab, V, path_vocab):
    revkv_vocab_dict = dict(zip(range(V), vocab))
    save_dict_to_json(revkv_vocab_dict, path_vocab)


######## utils for initiating S ########
def t2i(t, T, P):
    '''
    initiate S
    :param t: current period
    :param T: total period
    :param P: number of regimes
    :return: current regime
    '''
    l = float(T) / P
    for i in range(1, P + 1, 1):
        if t + 1 <= i * l:
            return i - 1


def initS(T, m):
    P = int(T / m)  # initial n_regimes
    S = [t2i(t, T, P) for t in range(T)]  # 等距法初始化
    return S


def initBeta(S, b0, b1):
    Sset = list(set(S))
    Sset.sort()
    initbetas = np.random.uniform(low=b0, high=b1, size=len(Sset))
    initbetas.sort()
    betaI = dict()
    for i in range(len(Sset)):
        betaI[Sset[i]] = initbetas[i]
    return betaI


def initZ(T, docs, D, N, K, V, vocab_dict):
    ntdk = [[[0] * K for _ in range(D[t])] for t in range(T)]
    ntkv = np.zeros(shape=(T, K, V), dtype=np.int64)
    ntd = [[np.sum(ntdk[t][d]) for d in range(D[t])] for t in range(T)]
    ntk = np.sum(ntkv, axis=-1)
    Ztdn = [[[0] * ntd_ for ntd_ in nt] for nt in N]
    Vtdn = [[[0] * ntd_ for ntd_ in nt] for nt in N]

    for t in range(T):
        for d in range(D[t]):
            for n in range(N[t][d]):
                z_vec = np.random.multinomial(1, [1.0 / K] * K)
                k = int(np.where(z_vec == 1)[0])
                v = vocab_dict[docs[t][d][n]]
                Ztdn[t][d][n] = k
                Vtdn[t][d][n] = v
                ntdk[t][d][k] += 1
                ntd[t][d] += 1
                ntkv[t][k][v] += 1
                ntk[t][k] += 1

    return Ztdn, Vtdn, ntdk, ntd, ntkv, ntk


######## utils for updating S ########
def cal_nii(S, t):
    i = S[t]
    nii = 0
    for tt in range(t - 1, -1, -1):
        if S[tt] == i:
            nii += 1
        else:
            break
    return nii


def cal_ni1i1(S, t):
    T = len(S)
    i1 = S[t]
    ni1i1 = 0
    for tt in range(t + 1, T, 1):
        if S[tt] == i1:
            ni1i1 += 1
        else:
            break
    return ni1i1


def add_list(lis1, lis2):
    return [a + b for a, b in zip(lis1, lis2)]


def acc_multiply(vec):
    return reduce(lambda x, y: x * y, vec)


def log_gamma(x):
    if x > 100:
        return sum([np.log(i) for i in range(1, int(x) + 1, 1)])
    else:
        return np.log(math.gamma(x))


def log_delta(vec):
    log_no = sum([log_gamma(v) for v in vec])
    log_deno = log_gamma(sum(vec))
    res = log_no - log_deno
    return res


# calculate log to avoid overflow error
def cal_log_pw(t, tt, S, betaI, V, K, ntkv):
    i = S[tt]
    beta_i = betaI[i]  # V-dim vec

    def map_k(k):
        return log_delta(add_list(ntkv[t][k], [beta_i] * V)) - log_delta([beta_i] * V)

    pool = Threadpool(processes=16)
    p = pool.map(map_k, range(K))
    pool.close()
    pool.join()
    return sum(p)


def update_st(S, t, alpha0, beta0, betaI, V, K, ntkv):
    nii = cal_nii(S, t - 1)
    ni1i1 = cal_ni1i1(S, t + 1)

    log_pt_i = np.log((nii + alpha0) / (nii + alpha0 + beta0))
    log_pt1_i = np.log(beta0 / (nii + 1 + beta0 + alpha0))
    log_pw_i = cal_log_pw(t, t - 1, S, betaI, V, K, ntkv)

    log_pt_i1 = np.log(beta0 / (nii + beta0 + alpha0))
    log_pt1_i1 = np.log((ni1i1 + alpha0) / (ni1i1 + beta0 + alpha0))
    log_pw_i1 = cal_log_pw(t, t + 1, S, betaI, V, K, ntkv)

    log_pi = log_pt_i + log_pt1_i + log_pw_i
    log_pi1 = log_pt_i1 + log_pt1_i1 + log_pw_i1

    # sample from b(pi_reg,1)
    try:
        pi = np.exp(log_pi)
        pi1 = np.exp(log_pi1)
        pi_reg = pi / (pi + pi1)
        u = np.random.rand(1)[0]
        if u < pi_reg:
            return S[t - 1]
        else:
            return S[t + 1]
    except RuntimeWarning:
        # except for exp error, choose max(pi, pi1) instead of sampling
        if log_pi >= log_pi1:
            return S[t - 1]
        else:
            return S[t + 1]


def update_s0(S, alpha0, beta0, betaI, V, K, ntkv):
    ns2s2 = cal_ni1i1(S, 1)
    log_pi = np.log(alpha0 / (alpha0 + beta0)) + np.log(beta0 / (beta0 + alpha0)) + \
             cal_log_pw(0, 0, S, betaI, V, K, ntkv)
    log_pi1 = np.log(beta0 / (alpha0 + beta0)) + np.log((ns2s2 + alpha0) / (ns2s2 + alpha0 + beta0)) + \
              cal_log_pw(0, 1, S, betaI, V, K, ntkv)
    try:
        pi1 = np.exp(log_pi1) / (np.exp(log_pi) + np.exp(log_pi1))
        u = np.random.rand(1)[0]
        if u < pi1:
            return S[1]
    except RuntimeWarning:
        if log_pi < log_pi1:
            return S[1]
    return S[0]


def update_sT(T, S, alpha0, beta0, betaI, V, K, ntkv):
    nsn1sn1 = cal_nii(S, T - 2)
    log_pi = np.log((nsn1sn1 + alpha0) / (nsn1sn1 + alpha0 + beta0)) + \
             cal_log_pw(T - 1, T - 2, S, betaI, V, K, ntkv)
    log_pi1 = np.log(beta0 / (nsn1sn1 + alpha0 + beta0)) + cal_log_pw(T - 1, T - 1, S, betaI, V, K, ntkv)

    try:
        pi = np.exp(log_pi) / (np.exp(log_pi) + np.exp(log_pi1))
        u = np.random.rand(1)[0]
        if u < pi:
            return S[T - 2]
    except RuntimeWarning:
        if log_pi >= log_pi1:
            return S[T - 2]
    return S[T - 1]


def updateS(T, S, alpha0, beta0, betaI, V, K, ntkv):
    # for t = 1 to T-2
    for t in range(1, T - 1, 1):
        # identify change points sequentialy
        if S[t - 1] != S[t + 1]:
            S[t] = update_st(S, t, alpha0, beta0, betaI, V, K, ntkv)

    # for t = 0
    if S[0] != S[1]:
        S[0] = update_s0(S, alpha0, beta0, betaI, V, K, ntkv)

    # for t = T-1
    if S[T - 1] != S[T - 2]:
        S[T - 1] = update_sT(T, S, alpha0, beta0, betaI, V, K, ntkv)
    return S


######## utils for updating nikv ########
def updateNikv(S, T, D, N, Ztdn, Vtdn, K, V):
    nikv = dict()
    nik = dict()
    for t in range(T):
        i = S[t]
        for d in range(D[t]):
            for n in range(N[t][d]):
                k = Ztdn[t][d][n]
                v = Vtdn[t][d][n]
                if i not in nikv:
                    nikv[i] = np.zeros(shape=(K, V))
                    nik[i] = np.zeros(shape=K)
                nikv[i][k][v] += 1
                nik[i][k] += 1
    return nikv, nik


######## utils for updating beta ########
def reflect(x, b2, epsilon):
    x = abs(x)
    x_reflected = 2 * b2 - x if x > b2 else x
    if x_reflected == 0:
        return x_reflected + 0.1 * epsilon
    return x_reflected


def proposeBeta(beta_ori, epsilon, b2):
    beta_star = reflect(np.random.uniform(beta_ori - epsilon, beta_ori + epsilon, 1)[0], b2, epsilon)
    return beta_star


def updateBetai(i, betaI, b2, epsilon, nikv, V, K):
    beta_ori = betaI[i]
    beta_star = proposeBeta(beta_ori, epsilon, b2)

    log_pi_star = sum([log_delta(add_list(nikv[i][k], [beta_star] * V)) - log_delta([beta_star] * V) for k in range(K)])
    log_pi_ori = sum([log_delta(add_list(nikv[i][k], [beta_ori] * V)) - log_delta([beta_ori] * V) for k in range(K)])

    if log_pi_star - log_pi_ori > 3:
        r = 1
    else:
        r = np.exp(log_pi_star - log_pi_ori)

    u = np.random.rand(1)[0]
    if u < r:
        return beta_star
    else:
        return beta_ori


def updateBeta(S, betaI, b2, epsilon, nikv, V, K):
    Sset = list(set(S))
    for i in Sset:
        betaI[i] = updateBetai(i, betaI, b2, epsilon, nikv, V, K)
    return betaI


######## utils for updating Z ########
def sampleZ(S, t, d, v, ndk, nikv, nik, K, V, betaI, alpha):
    i = S[t]
    p = [((nikv[i][k][v] + betaI[i]) / (nik[i][k] + V * betaI[i])) * (ndk[d][k] + alpha[k]) for k in range(K)]
    p_uni = [i / sum(p) for i in p]
    k_vec_new = np.random.multinomial(1, p_uni)
    k_new = int(np.where(k_vec_new == 1)[0])
    return k_new


######## utils for saving cps ########
def find_breaks(vec):
    breaks = []
    for i in range(1, len(vec)):
        if vec[i] != vec[i - 1]:
            breaks.append(i)
    return breaks


def saveCp(path_output, iter, S, betaI, true_tau, true_beta):
    breaks = find_breaks(S)
    n_cp = len(breaks)
    keys = list(set(S))
    keys.sort()
    betas = [betaI[kk] for kk in keys]
    with open(path_output, 'a') as output_cp:
        output_cp.write("{},{},{},{},{},{},{}\n". \
            format(iter, n_cp, ' '.join(map(str, breaks)), ' '.join(map(str, betas)),
                len(true_tau), ' '.join(map(str, true_tau)), ' '.join(map(str, true_beta))))

