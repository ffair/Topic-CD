
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

path_output = 'len.txt'
# path_file = 'phone.json'
path_file = 'reviews_Cell_Phones_and_Accessories_5.json/Cell_Phones_and_Accessories_5.json'
path_stopwords = 'stopwords.txt'
stopwords = {}.fromkeys([line.strip() for line in open(path_stopwords, 'r')])
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()
#qdict = defaultdict(dictypes)


n = 0
qlist=[]
#with gzip.open(path_file) as f:
with open(path_file) as f:
    for l in f:
        n += 1
        if n % 10000 == 0 and n >= 10000:
            print(n)
        try:
            js = eval(l.strip())
            datestr = js['reviewTime']
            q = date2Q(datestr)
            qlist.append(q)
        except:
            print("error!")

a=Counter(qlist)

with open(path_output,'w') as f_out:
    for q,dic in a.items():
        f_out.write('{},{}\n'.format(q,dic))
