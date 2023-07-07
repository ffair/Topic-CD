#-*- coding: utf-8 -*-
import nltk
import re
import datetime
from dateutil.relativedelta import relativedelta
import json
from collections import defaultdict
from collections import Counter
from functools import reduce


def extractYear(datestr):
    '''
    extract year from the origin datestr
    :param datestr: "02 28, 2014"
    :return:
    '''
    return int(datestr.split(', ')[1])


def extractMonth(datestr):
    return int(datestr.split(' ')[0]) - 1


def date2Q(datestr):
    ori = datetime.datetime.strptime("08 1, 2011", "%m %d, %Y")
    d = datetime.datetime.strptime(datestr, "%m %d, %Y")
    diff = relativedelta(d, ori)
    return int((diff.years * 12 + diff.months) / 1)


def date2mon(datestr):
    ori = datetime.datetime.strptime("05 1, 1996", "%m %d, %Y")
    d = datetime.datetime.strptime(datestr, "%m %d, %Y")
    diff = relativedelta(d, ori)
    return diff.years * 12 + diff.months


def date2week(datestr):
    d = datetime.datetime.strptime(datestr, "%m %d, %Y")
    week = datetime.datetime.strftime("%U", d)
    return int(week)


def delPuncDigits(line):
    # delEStr = string.punctuation + string.digits
    # r1 = '[' + delEStr + ']+'
    r1 = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789]+'
    filter_line = re.sub(r1, '', line).strip()
    return filter_line


# tokenize
def wordTokener(sent):  
    wordsInStr = nltk.word_tokenize(sent)
    return wordsInStr


# delete stop words
def delStopShortLow(sentence, stopwords):
    filter_sentence = [w.lower() for w in sentence if w not in stopwords and 3 <= len(w)]
    return filter_sentence


# stemming and lemmatization
def stemLemm(sentence, stemmer, wnl):
    line = [wnl.lemmatize(stemmer.stem(w)) for w in sentence]
    return line


# cleanLine
def cleanLine(line, stopwords, stemmer, wnl):
    l1 = delPuncDigits(line)
    l2 = wordTokener(l1)
    l3 = stemLemm(l2, stemmer, wnl)
    l4 = delStopShortLow(l3, stopwords)
    return l4


# tokenize
def wordTokener_v2(sent):  # 将单句字符串分割成词
    wordsInStr = nltk.word_tokenize(sent.lower())
    return wordsInStr


# delete stop words
def delStopShort(sentence, stopwords):
    filter_sentence = [w for w in sentence if w not in stopwords] # and 3 <= len(w)]
    return filter_sentence


def cleanLine_v2(line, stopwords, stemmer, wnl):
    l1 = wordTokener_v2(line)
    l2 = delStopShort(l1,stopwords)
    l3 = delPuncDigits(' '.join(l2)).split(' ')
    l4 = stemLemm(l3, stemmer, wnl)
    return l4


def cleanLine_v3(line, stopwords, stemmer, wnl):
    l1 = wordTokener_v2(line)
    l2 = delStopShort(l1,stopwords)
    l3 = [w for w in delPuncDigits(' '.join(l2)).split(' ') if len(w) >= 3]
    return l3



# merge dicts
def mergeDicts(dicts):
    dd = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            dd[key] += value
    return dd


# save dict to json
def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


# extract_top_vocab
def extract_top_vocab(dic, n_top):
    cnt = Counter(dic)
    top_vocab_tuple = cnt.most_common(n_top)
    top_vocab_list = [k for (k, v) in top_vocab_tuple]
    top_vocab_cnt_list = [v for (k, v) in top_vocab_tuple]

    n_words = sum(cnt.values())
    top_vocab_rate_list = [round(v / n_words, 4) for v in top_vocab_cnt_list]
    top_vocab_accrate = sum(top_vocab_rate_list)

    return top_vocab_list, top_vocab_rate_list, top_vocab_accrate
