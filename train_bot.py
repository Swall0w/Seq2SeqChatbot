# coding:utf-8
import MeCab
from model import Seq2Seq
import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda
from chainer import Variable
import numpy as np
import sys
import codecs
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
tagger = MeCab.Tagger('-Owakati')

def parse_sentence(sentence):
    parsed = []
    sentence = tagger.parse(sentence)[:-1]
    for surface in sentence.split(' '):
        parsed.append(surface)
    return parsed

def parse_file(filename):
    questions = []
    answers = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            sentences = line.split("\t")
            question = ["<start>"] + parse_sentence(sentences[0]) + ["<eos>"]
            answer = parse_sentence(sentences[1]) + ["<eos>"]
            questions.append(question)
            answers.append(answer)
    word2id = {"■": 0}
    id2word = {0: "■"}
    id = 1
    sentences = questions + answers
    for sentence in sentences:
        for word in sentence:
            if word not in word2id:
                word2id[word] = id
                id2word[id] = word
                id += 1
    return questions, answers, word2id, id2word

def sentence_to_word_id(split_sentences, word2id,gpu=-1):
    xp = cuda.cupy if gpu >=0 else np
    id_sentences = []
    for sentence in split_sentences:
        ids = []
        for word in sentence:
            id = word2id[word]
            ids.append(id)
        id_sentences.append(ids)
    return id_sentences


class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.iteration = 0

    def __next__(self):

        length = len(self.dataset[0])
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration

        batch_start_index = self.iteration * self.batch_size % length
        batch_end_index = min(batch_start_index + self.batch_size, length)

        questions = [self.dataset[0][batch_index] for batch_index in range(batch_start_index, batch_end_index)]
        answers = [self.dataset[1][batch_index]for batch_index in range(batch_start_index, batch_end_index)]

        self.iteration += 1

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(questions, answers))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset[0])

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.xp = cuda.cupy if self.device >= 0 else np

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        batch = train_iter.__next__()
        for question, answer in batch:
            x = self.xp.array(question,dtype=self.xp.int32)
            t = self.xp.array(answer,dtype=self.xp.int32)
            loss += optimizer.target(x, t)

        optimizer.target.cleargrads()       # Clear the paramer gradients
        loss.backward()                     # Backprop
        loss.unchain_backward()             # Truncate the graph
        optimizer.update()                  # Update the paramers

def main():
    gpu = -1
    result = 'result'
    epoch_n = 70 
    questions, answers, word2id, id2word = parse_file("data/separate/white.txt")
    ids_questions = sentence_to_word_id(questions, word2id=word2id,gpu=gpu)
    ids_answers = sentence_to_word_id(answers, word2id=word2id,gpu=gpu)

    model = Seq2Seq(len(word2id))
    xp = cuda.cupy if gpu >=0 else np
    if gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = ParallelSequentialIterator(dataset=(ids_questions, ids_answers), batch_size=1)

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch_n, 'epoch'),result)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['elapsed_time','epoch', 'iteration', 'main/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.snapshot())
    trainer.run()
    if gpu >= 0:
        model.to_cpu()
    chainer.serializers.save_npz("model.npz", model)
    json.dump(id2word, open("dictionary_i2w.json", "w"))
    json.dump(word2id, open("dictionary_w2i.json", "w"))

if __name__ == '__main__':
    main()
