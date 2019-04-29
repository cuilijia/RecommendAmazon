from gensim.models import Word2Vec
import json
from sklearn.decomposition import PCA
from matplotlib import pyplot


def train(file,Sg,Size,Window,Min_count,Workers,Iter):
    load_dict=[]
    with open(file,'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    sentences = []
    for i in range(len(load_dict[0])):
        print(load_dict[0][str(i)])
        sentences.append(load_dict[0][str(i)])

    model = Word2Vec(sentences,sg=Sg,size=Size,window=Window,min_count=Min_count,workers=Workers,iter=Iter)
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

def drawpic(size):
    model = Word2Vec.load('word2vec.model')
    # 基于2d PCA拟合数据
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # 可视化展示

    if size==0:
        pyplot.scatter(result[:, 0], result[:, 1])
    else:
        pyplot.scatter(result[:size, 0], result[:size, 1])
    words = list(model.wv.vocab)[:0]
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

# train("data/data1.json",Sg=0,Size=250,Window=5,Min_count=1,Workers=4,Iter=10)
# for i in range(2,11):
#     addtrain("data/data"+str(i)+".json")

# test(["4831", "13644", "14643"])

drawpic(0)
# 输入是画的size，为0是全部输出