# coding:utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import time
import sys
import argparse
import cPickle as pickle
import os

import RNN

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='data')
parser.add_argument('--data_file',      type=str,   default='Descartes')
parser.add_argument('--model_class',    type=str,   default='LSTM')
parser.add_argument('--model_file',     type=str,   default='latest.model')
parser.add_argument('--l1_size',        type=int,   default=100)
parser.add_argument('--l2_size',        type=int,   default=100)
parser.add_argument('--seed',           type=int,   default=1)
parser.add_argument('--deterministic',  type=bool,   default=False)
args = parser.parse_args()

np.random.seed(args.seed)


vocab = pickle.load(open('%s/%s_vocab.bin' % (args.data_dir, args.data_file), 'rb'))
train_data = pickle.load(open('%s/%s_train_data.bin' % (args.data_dir, args.data_file), 'rb'))

model_class = 'RNN.%s(%s,%s,%s)' % (args.model_class, len(vocab), args.l1_size, args.l2_size)
print model_class
rnn = eval(model_class)
model = L.Classifier(rnn)

checkpoint_dir = '%s_%s_%s_%s' % ( args.data_file, args.model_class, args.l1_size, args.l2_size )
serializers.load_npz(checkpoint_dir+'/'+args.model_file, model)

# vocabのキーと値を入れ替えたもの
ivocab = {}
for c, i in vocab.iteritems():
    ivocab[i] = c

for i in xrange(30):
    model.predictor.reset_state()
    index = np.random.randint(1, len(vocab))
    while index != 0:
        sys.stdout.write( ivocab[index].split("::")[0] )
        y = model.predictor(np.array([index], dtype=np.int32))
        probability = F.softmax(y)
        if args.deterministic:
            index = int(F.argmax(probability.data[0]).data)
            continue
        probability.data[0] /= sum(probability.data[0])
        try:
            index = np.random.choice(range(len(probability.data[0])), p=probability.data[0])
        except:
            print 'probability error'
            break
    print '\n=========='

print 'generated!'