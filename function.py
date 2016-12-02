# coding:utf-8
import re
import CaboCha

def remove_duplicates(x):
    y=[]
    for i in x:
        if i not in y:
            y.append(i)
    return y
def make_function_hiragana():
    re_katakana = re.compile(ur'[ァ-ヴ]')
    def hiragana(text):
        """ひらがな変換"""
        return re_katakana.sub(lambda x: unichr(ord(x.group(0)) - 0x60), text)
    return hiragana
hiragana = make_function_hiragana()


def get_chunk(sentence):
    c = CaboCha.Parser()

    tree = c.parse(sentence)
    line_a = tree.toString(CaboCha.FORMAT_LATTICE).splitlines()
    word_a = []
    kana_a = []
    chunk_a= []
    #print tree.toString(CaboCha.FORMAT_LATTICE)
    pre_line = '';
    for line in line_a:
        #英単語の場合強制的に文節
        if re.match(r"^\*",line) or re.match(r"EOS",line) or re.match(r'^[a-zA-Z0-9_]+$',pre_line):
            if len(word_a)!=0:
                chunk_a.append("".join(word_a)+"::"+"".join(kana_a));
                word_a = []
                kana_a = []
            if re.match(r"^\*",line) or re.match(r"EOS",line):
                continue
        data_a = line.split('	')
        word_a.append(data_a[0])
        feature_a = data_a[1].split(',')
        if len(feature_a)>7:
            kana_a.append(data_a[1].split(',')[7])
        #print line
        pre_line = data_a[0]
    chunk_a.append("EOS")
    return chunk_a
