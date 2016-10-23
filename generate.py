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
import function

import RNN

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='data')
parser.add_argument('--data_file',      type=str,   default='Descartes')
parser.add_argument('--model_class',    type=str,   default='PersonLSTM')
parser.add_argument('--model_file',     type=str,   default='latest.model')
parser.add_argument('--l1_size',        type=int,   default=100)
parser.add_argument('--l2_size',        type=int,   default=100)
parser.add_argument('--seed',           type=int,   default=1)
parser.add_argument('--deterministic',  type=bool,   default=False)
parser.add_argument('--gpu',            type=int,   default=-1)
parser.add_argument('--layer',            type=int,   default=3)

#person
parser.add_argument('--person_size',        type=int,   default=10)
args = parser.parse_args()

xp = cuda.cupy if args.gpu >= 0 else np
xp.random.seed(args.seed)

use_person = args.model_class=='PersonLSTM'

vocab = pickle.load(open('%s/%s_vocab.bin' % (args.data_dir, args.data_file), 'rb'))
train_data = pickle.load(open('%s/%s_train_data.bin' % (args.data_dir, args.data_file), 'rb'))

if use_person:
    person_data = pickle.load(open('%s/%s_person_data.bin' % (args.data_dir, args.data_file), 'rb'))
    person_unique_a = function.remove_duplicates(person_data)
    if 'unknown' not in person_unique_a:
        person_unique_a.append('unknown')

if args.model_class=='PersonLSTM':
    model_class = 'RNN.%s(%s,%s,%s,%s,%s,%i)' % (args.model_class, len(vocab),len(person_unique_a), args.l1_size,args.person_size, args.l2_size,int(args.layer))
else:
    model_class = 'RNN.%s(%s,%s,%s,%i)' % (args.model_class, len(vocab), args.l1_size, args.l2_size,int(args.layer))
print model_class
rnn = eval(model_class)
model = L.Classifier(rnn)
if args.gpu >= 0:
    print 'use GPU!'
    cuda.get_device(args.gpu).use()
    model.to_gpu()

checkpoint_dir = '%s_%s_%s_%s' % ( args.data_file, args.model_class, args.l1_size, args.l2_size )
serializers.load_npz(checkpoint_dir+'/'+args.model_file, model)

# vocabのキーと値を入れ替えたもの
ivocab = {}
for c, i in vocab.iteritems():
    ivocab[i] = c

#person list


for i in xrange(30):
    model.predictor.reset_state()
    index = xp.random.randint(1, len(vocab))
    while index != 0:
        sys.stdout.write( ivocab[index].split("::")[0] )
        y = model.predictor([xp.array([index], dtype=xp.int32),xp.array([2], dtype=xp.int32)])
        probability = F.softmax(y)
        if args.deterministic:
            index = int(F.argmax(probability.data[0]).data)
            continue
        probability.data[0] /= sum(probability.data[0])
        try:
            index = xp.random.choice(range(len(probability.data[0])), p=probability.data[0])
        except:
            print 'probability error'
            break
    print '\n=========='

print 'generated!'