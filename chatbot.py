# coding:utf-8
import MeCab
import codecs
from model import Seq2Seq
import chainer
import json
import sys
import io

class Chatbot:
    def __init__(self, dirname):
        self.dir = 'model/' + dirname + '/'
        self.dict_i2w = self.dir + 'dictionary_i2w.json'
        self.dict_w2i = self.dir + 'dictionary_w2i.json'
        self.modelname = self.dir + 'model.npz'

    def initialize(self):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.tagger = MeCab.Tagger('-Owakati')
        self.id2word = json.load(open(self.dict_i2w, "r"))
        self.id2word = {int(key): value for key, value in self.id2word.items()}
        self.word2id = json.load(open(self.dict_w2i, "r"))
        self.model = Seq2Seq(input_words=len(self.word2id), train=False)
        chainer.serializers.load_npz(self.modelname, self.model)

    def get_reply(self,message):
        try:
            parsed_sentence = []
            sentence = self.tagger.parse(message)[:-1]
            for surface in sentence.split(' '):
                parsed_sentence.append(surface)
            parsed_sentence = ["<start>"] + parsed_sentence + ["<eos>"]
            ids = []
            for word in parsed_sentence:
                if word in self.word2id:
                    id = self.word2id[word]
                    ids.append(id)
                else:
                    ids.append(0)
            ids_question = ids
            sentence = "".join(self.model.generate_sentence(ids_question, dictionary=self.id2word)).encode("utf-8")
            return sentence.decode('utf-8')
        except Exception as e:
            return e, '解析できませんでした。。。'


if __name__ == "__main__":
    main()
