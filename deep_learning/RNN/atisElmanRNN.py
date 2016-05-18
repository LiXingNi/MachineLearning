# _*_ coding:utf-8 _*_

import numpy as np
import theano
from theano import function
import theano.tensor as T
from elman import ElmanRnn
import cPickle

def loadATIS(filename):
    train_set, valid_set, test_set, dic = cPickle.load(open(filename))
    return train_set, valid_set, test_set, dic

def contextWindow(l,cs):
    assert cs % 2 == 1
    l = list(l)
    l_append = (cs / 2) * [-1]  + l + (cs / 2) * [-1]
    l_result = [ l_append[i : i + cs] for i in range(len(l))]
    return l_result

def shuffle(lol,seed):
    for l in lol:
        np.random.seed(seed)
        np.random.shuffle(l)

def minibatch(l, bs):
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def zeroOne(l1,l2):
    return T.sum(T.neq(l1,l2)) / len(l1)


def train(filename):
    s = {'lr': 0.06,
         'nhidden' : 100,
         'win'  : 7,
         'bs' : 9,
         'seed' : 345,
         'emb_dimension' : 100,
         'nepoch' : 50
         }
    train_set, valid_set, test_set, dic = loadATIS(filename)
    idx2words = dict([(v,k) for k, v in dic['words2idx'].iteritems()])
    idx2labels = dict([(v,k) for k, v in dic['labels2idx'].iteritems()])

    train_lex, train_ne, train_y = train_set
    test_lex, test_ne, test_y = test_set
    valid_lex, valid_ne, valid_y = valid_set

    vocabulary_num = len(idx2words)
    class_num = len(idx2labels)

    rnn = ElmanRnn(vocabulary_num, s['nhidden'],
                   class_num, s['emb_dimension'],
                   s['win'])
    f = open("./train_result","w")
    for e in xrange(s['nepoch']):
        f.writelines( str(e) +  "-"*70)
        shuffle([train_lex,train_ne,train_y],s['seed'])
        for i, line in enumerate(train_lex):
            cwords = contextWindow(line,s['win'])
            words =  map(lambda x : np.asarray(x).astype('int32'),
                         minibatch(cwords, s['bs']))
            labels = train_y[i]

            precision = 0.0

            for word_batch, last_word_label in zip(words, labels):
                precision += rnn.train(word_batch, last_word_label, s['lr'])
                rnn.normalize()
            print ("for epoch : %d. precision for line %d is %f"%(e, i,precision))

        for i, test_line in enumerate(test_lex):
            prediction = rnn.classifier(contextWindow(test_line,s['win']))
            print ("for epoch : %d . for sentence %d , the zero_one cross is : %f"%
                   (e,i, zeroOne(prediction,train_y[i])))
            s_org = " ".join([idx2words[j] for j in test_y[i]])
            s_pred = " ".join([idx2words[j] for j in prediction])
            f.writelines(s_org + " ---- " + s_pred)


    f.close()



if __name__ == "__main__":
    train("D:\\test.pkl")






