from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"],
             ["dog", "say", "woof"],
             ["dog", "say", "cui"],
             ["dog", "say", "log"],
             ["cat", "say", "li"],
             ["dog", "eat", "woof"]]

sentences2 = [["cat", "say", "me"],["cat", "say", "me"],["cat", "say", "love"],
              ["cat", "say", "me"],["cat", "say", "me"],["cat", "say", "love"]]
# vocab=["cat", "say", "meow","dog","me"]
# model = Word2Vec(min_count=1)
# model.build_vocab(sentences)
model = Word2Vec(sentences2,sg=0,size=250,window=5,workers=3,iter=20,min_count=1)

print(model.wv.most_similar(["cat", "say"]))

# model.build_vocab(sentences)  # prepare the model vocabulary
model.build_vocab(sentences,update=True)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
# self.word2vec_model.train(batch, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.iter)
# model = model.train(sentences2, total_examples=model.corpus_count, epochs=model.iter)
print(model.wv.most_similar(["cat", "say"]))