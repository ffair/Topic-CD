import numpy as np
import json
from collections import Counter

path_dict = '../1_docs.txt'
path_output = 'top10.txt'

d={}
list_before=[]
list_after=[]
with open(path_dict) as f:
    for line in f:
        a=line.split(',')
        q,word=int(a[0]),a[1]
        word=word.strip().split(' ')
        if q<=25:
            list_before.extend(word)
        else:
            list_after.extend(word)

list1 = [item for item in list_before if item not in list_after]
list2 = [item for item in list_after if item not in list_before]

print("old:",Counter(list1).most_common(50))
print()
print("new:",Counter(list2).most_common(50))
