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
import MeCab
import json
import re

import RNN

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='data')
parser.add_argument('--data_file',      type=str,   default='Descartes')
parser.add_argument('--model_class',    type=str,   default='PersonLSTM')
parser.add_argument('--model_file',     type=str,   default='epoch_100.model')
parser.add_argument('--l1_size',        type=int,   default=100)
parser.add_argument('--l2_size',        type=int,   default=100)
parser.add_argument('--seed',           type=int,   default=10)
parser.add_argument('--deterministic',  type=bool,   default=False)
parser.add_argument('--gpu',            type=int,   default=-1)
parser.add_argument('--layer',            type=int,   default=3)
parser.add_argument('--primetext',            type=str,   default='')
parser.add_argument('--reverse_model_file',           type=str,   default='')
parser.add_argument('--suggest_start_letters',           type=str,   default='')

#generate,suggest
parser.add_argument('--mode',           type=str,   default='normal')
parser.add_argument('--suggest_start',           type=str,   default='')

#person
parser.add_argument('--person_size',        type=int,   default=10)
parser.add_argument('--person',        type=str,   default="unknown")

parser.add_argument('--processor',  type=str,   default='cabocha')
args = parser.parse_args()

use_reverse = args.reverse_model_file!=''

xp = cuda.cupy if args.gpu >= 0  else np
xp.random.seed(args.seed)
mecab = MeCab.Tagger ("-Ochasen")

use_person = args.model_class=='PersonLSTM'

suffix = '' if args.processor=='mecab' else '_cabocha'
vocab = pickle.load(open('%s/%s%s_vocab.bin' % (args.data_dir, args.data_file,suffix), 'rb'))
train_data = pickle.load(open('%s/%s%s_train_data.bin' % (args.data_dir, args.data_file,suffix), 'rb'))

if use_person:
    person_data = pickle.load(open('%s/%s_person_data.bin' % (args.data_dir, args.data_file), 'rb'))
    person_unique_a = function.remove_duplicates(person_data)
    if 'unknown' not in person_unique_a:
        person_unique_a.append('unknown')

person_index = person_unique_a.index(args.person)
print "PERSON:"+args.person

if args.model_class=='PersonLSTM':
    model_class = 'RNN.%s(%s,%s,%s,%s,%s,%i)' % (args.model_class, len(vocab),len(person_unique_a), args.l1_size,args.person_size, args.l2_size,int(args.layer))
else:
    model_class = 'RNN.%s(%s,%s,%s,%i)' % (args.model_class, len(vocab), args.l1_size, args.l2_size,int(args.layer))
print model_class
rnn = eval(model_class)
model = L.Classifier(rnn)
if use_reverse:
    rnn = eval(model_class)
    model_reverse = L.Classifier(rnn)
if args.gpu >= 0:
    print 'use GPU!'
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    if use_reverse:
        model_reverse.to_gpu()

checkpoint_dir = '%s_%s_%s_%s%s' % ( args.data_file, args.model_class, args.l1_size, args.l2_size,suffix )
serializers.load_npz(checkpoint_dir+'/'+args.model_file, model)
if use_reverse:
    print 'REVERSE MODE!'
    checkpoint_dir = '%s_%s_%s_%s_reverse' % ( args.data_file, args.model_class, args.l1_size, args.l2_size )
    serializers.load_npz(checkpoint_dir+'/'+args.reverse_model_file, model_reverse)

# vocabのキーと値を入れ替えたもの
ivocab = {}
for c, i in vocab.iteritems():
    ivocab[i] = c

#prime text
first_index_a = []
if args.primetext!='':
    if args.processor == 'mecab':
        print 'use PRIME TEXT!'
        node = mecab.parseToNode(args.primetext)
        while node:
            if node.surface=="":
                node=node.next
                continue
            word = node.surface+"::"+node.feature
            first_index_a.append(vocab[word])
            node = node.next
    elif args.processor == 'cabocha':
        chunk_a = function.get_chunk(args.primetext)
        for chunk in chunk_a:
            if chunk!='EOS':
                first_index_a.append(vocab[chunk])


def get_index_a(_model,_first_index_a):
    _model.predictor.reset_state()
    _sentence_index_a = []
    if len(_first_index_a)>0:
        for index in _first_index_a:
            _sentence_index_a.append(index)
            _model.predictor([xp.array([index], dtype=xp.int32),xp.array([person_index], dtype=xp.int32)])
    else:
        index = np.random.randint(1, len(vocab))
        _sentence_index_a.append(index)
    counter=0
    while index != 0:
        y = _model.predictor([xp.array([index], dtype=xp.int32),xp.array([person_index], dtype=xp.int32)])
        probability = F.softmax(y)
        if args.deterministic:
            index = int(F.argmax(probability.data[0]).data)
            continue
        probability.data[0] /= sum(probability.data[0])
        try:
            index = xp.random.choice(range(len(probability.data[0])), p=probability.data[0])
            if index!=0:
                _sentence_index_a.append(index)
        except Exception as e:
            print 'probability error'
            break
        counter += 1

        #max length
        if counter>200:
            break
    return _sentence_index_a

def get_suggest_words(_model,_first_index_a,limit):
    global ivocab
    _model.predictor.reset_state()
    _sentence_index_a = []
    if len(_first_index_a)>0:
        for index in _first_index_a:
            _sentence_index_a.append(index)
            _model.predictor([xp.array([index], dtype=xp.int32),xp.array([person_index], dtype=xp.int32)])
    else:
        index = np.random.randint(1, len(vocab))
        _sentence_index_a.append(index)
    y = _model.predictor([xp.array([index], dtype=xp.int32),xp.array([person_index], dtype=xp.int32)])
    probability = F.softmax(y)
    probability.data[0] /= sum(probability.data[0])
    probability = probability.data[0]
    _key_a = np.argsort(probability)[::-1]
    _value_a = np.sort(probability)[::-1]
    result_a = []
    pattern = re.compile(r"^"+args.suggest_start_letters)
    for i,key in enumerate(_key_a):
        _data = ivocab[key].split("::")
        _data[1] = _data[1].split(',')
        if args.suggest_start_letters!='':
            if pattern.match(_data[0])==None and (len(_data[1])<8 or pattern.match(function.hiragana(_data[1][7].decode('utf-8')).encode('utf-8'))==None):
                continue

        result_a.append({
            'index':key,
            'surface':_data[0],
            'mecab':_data[1],
            'profitability':str(_value_a[i])
        })
        if len(result_a)==limit:
            break

    return result_a


if args.mode=='suggest':
    words_a = get_suggest_words(model,first_index_a,30)
    for data in words_a[0:5]:
        _tmp_index_a = first_index_a[::]
        _tmp_index_a.append(data['index'])
        continue
        sentence_index_a = get_index_a(model,_tmp_index_a)
        sentence = ''
        for index in sentence_index_a:
            sentence += ivocab[index].split("::")[0]
        data['sentence']= sentence

    print json.dumps(words_a)
else:
    print '\n-=-=-=-=-=-=-=-'
    for i in xrange(10):
        sentence_index_a = get_index_a(model,first_index_a)
        #reverse
        if use_reverse and args.primetext:
            sentence_index_a.reverse()
            sentence_index_a = get_index_a(model_reverse,sentence_index_a)
            sentence_index_a.reverse()
            pass


        for index in sentence_index_a:
            sys.stdout.write( ivocab[index].split("::")[0] )
        print '\n-=-=-=-=-=-=-=-'

    #print 'generated!'