import numpy as np
import json
from collections import Counter

path_dict = '../1_docs.txt'
path_output1 = 'old.txt'
path_output2='new.txt'

d={}
list_before=[]
list_after=[]
with open(path_dict) as f:
    for line in f:
        a=line.split(',')
        q,word=int(a[0]),a[1]
        word=word.strip().split(' ')
        if q<=15:
            list_before.extend(word)
        else:
            list_after.extend(word)

new=list(set(list_after)-set(list_before))
old=list(set(list_before)-set(list_after))



#print("消失的词:",' '.join(list(set(list_before)-set(list_after))))
list1 = [item for item in list_before if item in old]
list2 = [item for item in list_after if item in new]
print("old:",len(Counter(list1)))
print("new:",len(Counter(list2)))
