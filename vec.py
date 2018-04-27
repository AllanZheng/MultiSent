# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors



word_vectors = KeyedVectors.load_word2vec_format("vec/wiki.multi.ar.vec", binary=False)
word_vectors1 = KeyedVectors.load_word2vec_format("vec/wiki.multi.fr.vec", binary=False)
print(word_vectors1.size())
print(word_vectors.size())
'''
    vocab, vec = torchwordemb.load_word2vec_text("vec/wiki.multi.fr.vec")
    vocab1, vec1= torchwordemb.load_word2vec_text("vec/wiki.multi.ar.vec")
    x=torch.cat((vec,vec1))
    vocab.update(vocab1)
    return vec,vocab
'''




