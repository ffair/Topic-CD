
from collections import defaultdict
from collections import Counter
import os

path_dir = ''
path_out = os.path.join(path_dir,'stat_n_docs.txt')
path_out_2 = os.path.join(path_dir,'stat_n_vocabs.txt')
path_out_3 = os.path.join(path_dir,'stat_top100.txt')
path_docs = '../1_docs.txt'

cnt_mon = Counter()
cnt_vocab = defaultdict(Counter)

with open(path_docs) as f:
    for l in f:
        mon, doc = l.strip().split(',')
        cnt_mon.update([mon]) 
        cnt_vocab[mon].update(doc.split(' '))


mons = list(cnt_vocab.keys())
mons.sort()        


with open(path_out, 'w') as f1:
    f1.write("T,n_docs\n")
    # for k,v in cnt_mon.items():
    for mon in mons:
        m = int(mon)
        f1.write("{},{}\n".format(m,cnt_mon[mon]))


with open(path_out_3, 'w') as f3:
    f3.write("T,vocab,cnt,rate\n")
    with open(path_out_2, 'w') as f2:
        f2.write("T,n_vocabs\n")
#        for mon, dic in cnt_vocab.items():
        for mon in mons:
            dic = cnt_vocab[mon]
            m = int(mon) 
            f2.write("{},{}\n".format(m, len(dic)))
            top100 = dic.most_common(100)
            total_words = float(sum(dic.values()))
            for vocab, cnt in top100:
                f3.write("{},{},{},{}\n".format(m,vocab,cnt,cnt/total_words))

        

