# coding:utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from random import shuffle

import sys
import argparse
import cPickle as pickle
import os
import time
import function

import RNN

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default='data')
parser.add_argument('--data_file',      type=str,   default='Descartes')
parser.add_argument('--model_class',    type=str,   default='PersonLSTM')
parser.add_argument('--l1_size',        type=int,   default=100)
parser.add_argument('--l2_size',        type=int,   default=100)
parser.add_argument('--batch_size',     type=int,   default=5)
parser.add_argument('--epochs',         type=int,   default=100)
parser.add_argument('--start_epoch',    type=int,   default=0)
parser.add_argument('--save_interval',  type=int,   default=10)
parser.add_argument('--seed',           type=int,   default=1)
parser.add_argument('--gpu',            type=int,   default=-1)
parser.add_argument('--layer',            type=int,   default=3)
parser.add_argument('--reverse',            type=int,   default=0)
parser.add_argument('--processor',  type=str,   default='mecab')

#person
parser.add_argument('--person_size',        type=int,   default=10)

args = parser.parse_args()
use_person = args.model_class=='PersonLSTM'


xp = cuda.cupy if args.gpu >= 0 else np
xp.random.seed(args.seed)

# モデル保存用フォルダ
suffix = '' if args.processor=='mecab' else '_cabocha'
if args.reverse:
    checkpoint_dir = '%s_%s_%s_%s%s_reverse' % ( args.data_file, args.model_class, args.l1_size, args.l2_size,suffix )
else:
    checkpoint_dir = '%s_%s_%s_%s%s' % ( args.data_file, args.model_class, args.l1_size, args.l2_size,suffix )
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


vocab = pickle.load(open('%s/%s%s_vocab.bin' % (args.data_dir, args.data_file,suffix), 'rb'))
train_data = pickle.load(open('%s/%s%s_train_data.bin' % (args.data_dir, args.data_file,suffix), 'rb'))
train_data_len = len(train_data)

if args.reverse:
    print 'REVERSE MODE!'
    for i,data in enumerate(train_data):
        data = data[::-1]
        data = data[1:]
        data = np.append(data,0)
        train_data[i] = data

if use_person:
    print 'use PERSON!'
    person_data = pickle.load(open('%s/%s%s_person_data.bin' % (args.data_dir, args.data_file,suffix), 'rb'))
    person_index_a = []
    person_unique_a = function.remove_duplicates(person_data)
    if 'unknown' not in person_unique_a:
        person_unique_a.append('unknown')
    for person in person_data:
        person_index_a.append(person_unique_a.index(person))

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

# すでにあるモデルから学習を始める場合
if args.start_epoch > 0:
    serializers.load_npz(checkpoint_dir+'/epoch_%d.model' % (args.start_epoch), model)

optimizer = optimizers.SGD()
optimizer.setup(model)



def align_length(seq_list):
    global xp
    # 長さを揃えるため max_length に合わせて -1 で埋める
    max_length = 0
    for seq in seq_list:
        length = len(seq)
        if length > max_length:
            max_length = length
    seq_batch = [ np.full((max_length), -1, dtype=np.int32) for i in xrange(len(seq_list)) ]
    for i, data in enumerate(seq_list):
        seq_batch[i][:len(data)] = seq_list[i]
    return xp.array(seq_batch,dtype=xp.int32)

def compute_loss(seq_batch,person_list=None):
    loss = 0
    counter=0;
    for cur_word, next_word in zip(seq_batch.T, seq_batch.T[1:]):
        counter+=1
        if use_person:
            loss += model([cur_word,person_list], next_word)
        else:
            loss += model(cur_word, next_word)
    print "loss:"+str(loss.data/counter)
    return loss
def shuffle_in_unison(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1_shuf,list2_shuf



for epoch in xrange(args.start_epoch+1, args.epochs+1):
    print 'epoch %d/%d' % (epoch, args.epochs)
    start = time.time()
    #shuffle
    if use_person:
        train_data,person_index_a = shuffle_in_unison(train_data,person_index_a)
    else:
        np.random.shuffle(train_data)

    for i in xrange(0, train_data_len, args.batch_size):
        sys.stdout.write( '%d/%d\r' % (i, train_data_len) )
        sys.stdout.flush()
        model.predictor.reset_state()
        #print [ [(i+j)%train_data_len] for j in xrange(args.batch_size) ]
        seq_list = [ train_data[(i+j)%train_data_len] for j in xrange(args.batch_size) ]
        seq_batch = align_length(seq_list)
        if use_person:
            person_list = xp.array([ person_index_a[(i+j)%train_data_len] for j in xrange(args.batch_size) ], dtype=np.int32)
            optimizer.update(compute_loss, seq_batch,person_list)
        else:
            optimizer.update(compute_loss, seq_batch)

    if epoch % args.save_interval == 0:
        serializers.save_npz(checkpoint_dir+'/epoch_%d.model' % (epoch), model)

    print 'epoch %d end.' % (epoch)
    elapsed_time = time.time() - start
    print ("epoch_time:{0}".format(elapsed_time)) + "[sec]"


serializers.save_npz(checkpoint_dir+'/latest.model', model)
print '\ntrained!'