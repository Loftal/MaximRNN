# coding:utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

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
parser.add_argument('--l1_size',        type=int,   default=100)
parser.add_argument('--l2_size',        type=int,   default=100)
parser.add_argument('--batch_size',     type=int,   default=5)
parser.add_argument('--epochs',         type=int,   default=100)
parser.add_argument('--start_epoch',    type=int,   default=0)
parser.add_argument('--save_interval',  type=int,   default=10)
parser.add_argument('--seed',           type=int,   default=1)
args = parser.parse_args()

np.random.seed(args.seed)

# モデル保存用フォルダ
checkpoint_dir = '%s_%s_%s_%s' % ( args.data_file, args.model_class, args.l1_size, args.l2_size )
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


vocab = pickle.load(open('%s/%s_vocab.bin' % (args.data_dir, args.data_file), 'rb'))
train_data = pickle.load(open('%s/%s_train_data.bin' % (args.data_dir, args.data_file), 'rb'))
train_data_len = len(train_data)


model_class = 'RNN.%s(%s,%s,%s)' % (args.model_class, len(vocab), args.l1_size, args.l2_size)
print model_class
rnn = eval(model_class)
model = L.Classifier(rnn)

# すでにあるモデルから学習を始める場合
if args.start_epoch > 0:
    serializers.load_npz(checkpoint_dir+'/epoch_%d.model' % (args.start_epoch), model)

optimizer = optimizers.SGD()
optimizer.setup(model)



def align_length(seq_list):
    # 長さを揃えるため max_length に合わせて -1 で埋める
    max_length = 0
    for seq in seq_list:
        length = len(seq)
        if length > max_length:
            max_length = length
    seq_batch = [ np.full((max_length), -1, dtype=np.int32) for i in xrange(len(seq_list)) ]
    for i, data in enumerate(seq_list):
        seq_batch[i][:len(data)] = seq_list[i]
    return np.array(seq_batch)

def compute_loss(seq_batch):
    loss = 0
    for cur_word, next_word in zip(seq_batch.T, seq_batch.T[1:]):
        loss += model(cur_word, next_word)
    return loss



for epoch in xrange(args.start_epoch+1, args.epochs+1):
    print 'epoch %d/%d' % (epoch, args.epochs)
    np.random.shuffle(train_data)
    for i in xrange(0, train_data_len, args.batch_size):
        sys.stdout.write( '%d/%d\r' % (i, train_data_len) )
        sys.stdout.flush()
        model.predictor.reset_state()
        #print [ [(i+j)%train_data_len] for j in xrange(args.batch_size) ]
        seq_list = [ train_data[(i+j)%train_data_len] for j in xrange(args.batch_size) ]
        seq_batch = align_length(seq_list)
        optimizer.update(compute_loss, seq_batch)
    
    if epoch % args.save_interval == 0:
        serializers.save_npz(checkpoint_dir+'/epoch_%d.model' % (epoch), model)
    
    print 'epoch %d end.' % (epoch)


serializers.save_npz(checkpoint_dir+'/latest.model', model)
print '\ntrained!'