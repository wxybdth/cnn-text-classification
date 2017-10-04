import re
import codecs
import numpy as np
import random

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9,!?\'\`]",'',string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(pos_data, neg_data):
    pos = list(codecs.open(pos_data,encoding='utf8').readlines())
    pos = [string.strip() for string in pos]
    neg = list(codecs.open(neg_data,encoding='utf8').readlines())
    neg = [string.strip() for string in neg]
    pos_clean = [clean_str(string) for string in pos]
    neg_clean = [clean_str(string) for string in neg]

    data = pos_clean.extend(neg_clean)

    pos_label = [[1,0] for i in range(len(pos)) ]
    neg_label = [[0,1] for i in range(len(neg))]

    label = pos.extend(neg_label)

    return  data, label
def generate_batch(batch_size,data,label,num_epoch,shuffle=True):
    data = np.array(data)
    label = np.array(label)
    data_size = len(data)
    num_batch = int((data_size )/batch_size)
    for epoch in num_epoch:
        if shuffle == True:
            shuffle_index = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_index]
            shuffled_lable = label[shuffle_index]
        else:
            shuffled_data = data
            shuffled_lable = label
        for i in range(num_batch):
            strat_index = i * batch_size
            end_index = (i+1) * batch_size
            yield shuffled_data[strat_index, end_index],\
                  shuffled_lable[strat_index,end_index]




pos_data = 'data/rt-polaritydata/rt-polarity.pos'
neg_data = 'data/rt-polaritydata/rt-polarity.neg'
pos, neg, pos_label, neg_lable = load_data_and_labels(pos_data, neg_data)




