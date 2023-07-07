import numpy as np
import json
from collections import Counter

path_dict = '../1_docs.txt'
path_output = 'top50_word.txt'

d={}
with open(path_dict) as f:
    for line in f:
        a=line.split(',')
        q,word=a[0],a[1]
        word=word.strip().split(' ')
        if q not in d.keys():
            d.update({q:Counter(word)})
        else:
            d[q].update(word)

d=dict(sorted(d.items(), key=lambda e:int(e[0]), reverse=False))
for q in d.keys():
    d[q]=d[q].most_common(50)

with open(path_output,'w') as f_out:
    for q,dic in d.items():    
        f_out.write('{},{}\n'.format(q,' '.join(list(dict(dic).keys()))))

