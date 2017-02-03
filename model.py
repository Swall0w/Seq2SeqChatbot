# coding:utf-8
import chainer.links as L
import chainer.functions as F
import chainer
import numpy as np
from chainer import reporter
from chainer import Variable
from chainer import cuda

class Seq2Seq(chainer.Chain):
    def __init__(self, input_words,n_units=300, train=True,gpu=-1):
        super(Seq2Seq, self).__init__(
                embed=L.EmbedID(input_words, n_units),  
                l1=L.LSTM(n_units, n_units),         
                l2=L.LSTM(n_units, n_units),        
                l3=L.Linear(n_units, input_words)  
        )
        self.train = train
        self.gpu = gpu

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, sentence, target_sentence):
        self.reset_state()
        encode_vector = self.encode(sentence=sentence)
        self.loss = None
        self.loss = self.decode(vector=encode_vector, targer_sentence=target_sentence)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def encode(self, sentence):
        xp = cuda.cupy if self.gpu >= 0 else np
        c = None
        for word in sentence:
            x = xp.array([word], dtype=xp.int32)
            x = chainer.Variable(x)
            h = F.tanh(self.embed(x))
            c = self.l1(h)
        return c

    def decode(self, vector=None, targer_sentence=None, dictionary=None):
        xp = cuda.cupy if self.gpu >= 0 else np
        loss = 0
        if self.train:
            for index, target_word in enumerate(targer_sentence):
                if index == 0:
                    j = F.tanh(self.l2(vector))
                    pred_word = self.l3(j)
                else:
                    j = F.tanh(self.l2(j))
                    pred_word = self.l3(j)
                x = xp.array([target_word], dtype=xp.int32)
                x = chainer.Variable(x)
                loss += F.softmax_cross_entropy(pred_word,x)
            return loss
        else:
            gen_sentence = []
            cnt = 0
            while True:
                if cnt == 0:
                    j = F.tanh(self.l2(vector))
                    pred_word = self.l3(j)
                else:
                    j = F.tanh(self.l2(j))
                    pred_word = self.l3(j)
                id = np.argmax(pred_word.data)
                cnt += 1
                word = dictionary[id]
                if word == "<eos>":
                    return gen_sentence

                gen_sentence.append(word)
                if cnt == 100:
                    break
            return gen_sentence

    def generate_sentence(self, sentence, dictionary):
        self.reset_state()
        encode_vector = self.encode(sentence=sentence)
        return self.decode(vector=encode_vector, dictionary=dictionary)

