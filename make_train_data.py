# coding:utf-8
import numpy as np
import MeCab

import argparse
import cPickle as pickle
import os
import codecs

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   type=str,   default='data')
parser.add_argument('--data_file',  type=str,   default='Descartes')
args = parser.parse_args()


# テキストファイルを行ごとのリストとして読み込み
input_txt = '%s/%s.txt' % (args.data_dir, args.data_file)
f = codecs.open(input_txt, 'rb', 'utf-8')
lines = f.readlines()

# 行ごとに形態素に分解
m = MeCab.Tagger ("-Ochasen")
words_by_lines = []
for line in lines:
    words = []
    node = m.parseToNode(line.encode("utf-8"))
    node = node.next #先頭は「::BOS/EOS,*,*,*,*,*,*,*,*」なので飛ばす
    while node:
        word = node.surface+"::"+node.feature
        words.append(word)
        node = node.next
    words_by_lines.append(words)

# vocabを作る
vocab = {}
vocab["::BOS/EOS,*,*,*,*,*,*,*,*"] = 0 #文末を表す「::BOS/EOS,*,*,*,*,*,*,*,*」は「0」にしておく。
for words_by_line in words_by_lines:
    for word in words_by_line:
        if word not in vocab:
            vocab[word] = len(vocab)

# 行ごとにvocabで表現
dataset = []
max_length = 0
for i, words_by_line in enumerate(words_by_lines):
    length = len(words_by_line)
    if length > max_length:
        max_length = length
    datasetline = np.ndarray((length), dtype=np.int32)
    for j, word in enumerate(words_by_line):
        datasetline[j] = vocab[word]
    dataset.append(datasetline)


print 'line num:', len(dataset)
print 'line_max_length:', max_length
print 'vocab size:', len(vocab)
pickle.dump(vocab, open('%s/%s_vocab.bin' % (args.data_dir, args.data_file), 'wb'))
pickle.dump(dataset, open('%s/%s_train_data.bin' % (args.data_dir, args.data_file), 'wb'))