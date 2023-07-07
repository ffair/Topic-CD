import numpy as np
import json
from collections import Counter
import jieba,os,re
import gensim
from gensim import corpora, models, similarities

path_dict = '../1_docs.txt'
#path_dict = 'testlda.txt'
path_output1 = 'old.txt'
path_output2='new.txt'

d={}
list_before=[]
list_before1=[]
list_after1=[]
list_after=[]
list_all=[]
with open(path_dict) as f:
    for line in f:
        a=line.split(',')
        q,word=int(a[0]),a[1]
        word=word.strip().split(' ')
        list_all.extend(word)

a=set(list_all)
print(len(a))
