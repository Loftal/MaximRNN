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
    chunk_a= []
    #print tree.toString(CaboCha.FORMAT_LATTICE)
    for line in line_a:
        if re.match(r"^\*",line) or re.match(r"EOS",line):
            if len(word_a)!=0:
                chunk_a.append("".join(word_a));
                word_a = []
            continue
        data_a = line.split('	')
        word_a.append(data_a[0])
        #print line
    return chunk_a
