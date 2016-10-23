# coding:utf-8
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class LSTM(Chain):
    def __init__(self, n_vocab, l1_units, l2_units,layer):
        self.layer = layer
        if layer==2:
            super(LSTM, self).__init__(
                embed = L.EmbedID(n_vocab, l1_units, ignore_label=-1),
                l1=L.LSTM(l1_units, l2_units),
                l2=L.Linear(l2_units, n_vocab)
            )
        elif layer==3:
            super(LSTM, self).__init__(
                embed = L.EmbedID(n_vocab, l1_units, ignore_label=-1),
                l1=L.LSTM(l1_units, l2_units),
                l2=L.LSTM(l2_units, l1_units),
                l3=L.Linear(l1_units, n_vocab),
            )

    def reset_state(self):
        self.l1.reset_state()
        if self.layer==3:
            self.l2.reset_state()

    def __call__(self, x):
        if self.layer==2:
            h0 = self.embed(x)
            h1 = self.l1(h0)
            y = self.l2(h1)
        elif self.layer==3:
            h0 = self.embed(x)
            h1 = self.l1(h0)
            h2 = self.l2(h1)
            y = self.l3(h2)
        return y

class PersonLSTM(Chain):
    def __init__(self, n_vocab, n_person, l1_units, person_units, l2_units,layer):
        self.layer = layer
        super(PersonLSTM, self).__init__(
                embed = L.EmbedID(n_vocab, l1_units, ignore_label=-1),
                embed_person = L.EmbedID(n_person, person_units, ignore_label=-1),
                l1=L.LSTM(l1_units+person_units, l2_units),
                l2=L.LSTM(l2_units, l1_units),
                l3=L.Linear(l1_units, n_vocab),
            )

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        word_list = x[0]
        person_list = x[1]
        #単語ベクトル
        h0_vocab = self.embed(word_list)
        #話者ベクトル
        h0_person= self.embed_person(person_list)
        h0 = F.concat((h0_vocab, h0_person), axis=1)
        h1 = self.l1(h0)
        h2 = self.l2(h1)
        y = self.l3(h2)
        return y


class SimpleRNN(Chain):
    def __init__(self, n_vocab, l1_units, l2_units):
        super(ElmanNet, self).__init__(
            embed = L.EmbedID(n_vocab, l1_units, ignore_label=-1),
            l1=L.Linear(l1_units, l2_units),
            l1_h=L.Linear(l2_units, l2_units),
            l2=L.Linear(l2_units, n_vocab)
        )
    
    def reset_state(self):
        self.state = None

    def __call__(self, x):
        if self.state is None:
            self.state = np.zeros((x.shape[0],self.l1_h.W.shape[0]), dtype=np.float32)
        h0 = self.embed(x)
        self.state = F.sigmoid(self.l1(h0) + self.l1_h(self.state))
        y = self.l2(self.state)
        return y



