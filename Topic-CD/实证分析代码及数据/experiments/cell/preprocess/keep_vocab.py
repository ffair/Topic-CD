import os
import sys
from collections import Counter
import json
import numpy as np
path_vocab = 'vocab.txt'
path_dict = 'q132.json'

with open(path_dict) as f:
    qdict = json.load(f)

total_vocab = Counter()
for q,dic in qdict.items():
    if int(q) >= 0 and int(q) <= 35:
        total_vocab.update(dic[1])

print(len(total_vocab))
print('# cnt == 1: ',len([k for k,v in total_vocab.items() if v == 1]))
print('# cnt <= 3: ',len([k for k,v in total_vocab.items() if v <= 3]))
print('# cnt <= 5: ',len([k for k,v in total_vocab.items() if v <= 5]))
print('# cnt <= 10: ',len([k for k,v in total_vocab.items() if v <= 10]))

vocabs = [k for k,v in total_vocab.items() if v > 5]
print("total vocab kept: ".format(len(vocabs)))
with open(path_vocab,'w') as f:
    for v in vocabs:
        f.write("{}\n".format(v))

