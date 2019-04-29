from gensim.models import Word2Vec
import gensim.models.word2vec as w2v
import json

def train(file):
    load_dict=[]
    with open(file,'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    sentences = []
    for i in range(len(load_dict[0])):
        print(load_dict[0][str(i)])
        sentences.append(load_dict[0][str(i)])

    model = Word2Vec(sentences,sg=0,size=250,window=5,min_count=1,workers=4,iter=10)
    model.save('word2vec.model')

def addtrain(updatefile):
    load_dict=[]
    with open(updatefile,'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    sentences = []
    for i in range(len(load_dict[0])):
        sentences.append(load_dict[0][str(i)])

    model = Word2Vec.load('word2vec.model')

    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
    model.save('word2vec.model')
    print("Increasing learning complited!")

def test(productidlist):
    model = Word2Vec.load('word2vec.model')

    print(str(productidlist)+"-->"+str(model.wv.most_similar(productidlist)))

# train("data/data1.json")
# for i in range(2,11):
#     addtrain("data/data"+str(i)+".json")


test(["4831", "13644", "14643", "7826", "12864"])

