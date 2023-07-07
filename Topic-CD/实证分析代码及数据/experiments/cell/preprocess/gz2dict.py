# -*- coding: utf-8 -*-

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from collections import defaultdict
import os
# from utils_pre import extractYear
from utils_pre import date2mon
# from utils_pre import extractMonth
from utils_pre import date2Q
from utils_pre import cleanLine
from utils_pre import cleanLine_v2
from utils_pre import cleanLine_v3
from utils_pre import save_dict_to_json
import gzip
from collections import Counter

path_output = 'q132.json'
# path_file = 'phone.json'
path_file = 'reviews_Cell_Phones_and_Accessories_5.json/Cell_Phones_and_Accessories_5.json'
path_stopwords = 'stopwords.txt'
stopwords = {}.fromkeys([line.strip() for line in open(path_stopwords, 'r')])
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()


# get mon_dict
def dictypes():
    cnt1 = Counter()
    cnt2 = Counter()
    l = []
    d = defaultdict(list)
    return ([cnt1, cnt2, l, d])


qdict = defaultdict(dictypes)

print("start!")

qlist=[]
n = 0
#with gzip.open(path_file) as f:
with open(path_file) as f:
    for l in f:
        n += 1
        if n % 10000 == 0 and n >= 10000:
            print(n)
        try:
            js = eval(l.strip())
            datestr, text, pid = js['reviewTime'], js['reviewText'], js['asin']
            q = date2Q(datestr)
            if q>=0 and q<=35:
                qlist.append(q)
                doc_before = text.strip().split(' ')
                doc_after = cleanLine_v3(text, stopwords, stemmer, wnl)
                vocab_dic_before = Counter(doc_before)
                vocab_dic_after = Counter(doc_after)

                qdict[q][0].update(vocab_dic_before)  # counter
                qdict[q][1].update(vocab_dic_after)  # counter
                qdict[q][2].append(len(doc_after))  # list
                qdict[q][3][pid].append(doc_after)  # dict, with list as ele
        except Exception as e:
            print("error!", e, "\n", l)

print(dict(Counter(qlist)))
# save mon_dict
save_dict_to_json(qdict, path_output)
print("end!")
