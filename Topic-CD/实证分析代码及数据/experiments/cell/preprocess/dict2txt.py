import numpy as np
import json
mode = 'sub'
path_vocab = 'vocab.txt'
path_dict = 'q132.json'
path_output = '../1_docs.txt'.format(mode)

vocab = {}.fromkeys([line.strip() for line in open(path_vocab, 'r')])

with open(path_dict) as f:
    qdict = json.load(f)

print("T :", len(qdict))
qs = list(qdict.keys())
qs.sort()
print("total T: ", qs)

with open(path_output, 'w') as f_out:
    for q,dic in qdict.items():
        q = int(q)
        # if mon >= 170 and mon <= 216:
        if q >= 0 and q <= 35:
            docs = dic[3].values()
            #print("docs:",docs)
            if mode == 'all':
                docs_chosen = [[w for w in rev if w in vocab] for p in docs for rev in p if len(rev)>20]
                #print("docs_chosen:",docs_chosen)
            else:
                # choose top20% longest docs for each q
                docs_1 = [[w for w in rev if w in vocab] for p in docs for rev in p if len(rev)>20]
                n_docs_chosen = int(len(docs_1) * 0.2)
                print("q:{}, n_docs:{}, n_docs_chosen:{}".format(q, len(docs_1), n_docs_chosen))
                doc_len = np.array([len(doc) for doc in docs_1])
                top500_idx = doc_len.argsort()[(-1)*n_docs_chosen:][::-1].tolist()
                docs_chosen = [docs_1[idx] for idx in top500_idx]

            for d in docs_chosen:
                f_out.write("{},{}\n".format(q, ' '.join(d)))
