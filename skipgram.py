from gensim.models import Word2Vec
import gensim.models.word2vec as w2v
import json

def train(file):
    load_dict=[]
    with open(file,'r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    sentences = []
    # sentences = [["i","love","haha","tree"],["i","love","tree","haha"],["love","flawer","hoho"],["you","love","flawer","hoho"]]
    for i in range(len(load_dict[0])):
    # for i in range(5):
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
        # print(load_dict[0][str(i)])
        sentences.append(load_dict[0][str(i)])

    model = Word2Vec.load('word2vec.model')

    model.build_vocab(sentences, update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
    model.save('word2vec.model')
    print("Increasing learning complited!")

def test(productid):
    model = Word2Vec.load('word2vec.model')
    # for i in range(5):
    print(str(productid)+"-->"+str(model.wv.most_similar([str(productid)])))
        # , sentences[i][2], sentences[i][3], sentences[i][4]]

# train("datageneral/data0.json")
# for i in range(1,11):
#     addtrain("datageneral/data"+str(i)+".json")
addtrain("datageneral/data8.json")
test("18305")
# 5160-->[('14428', 0.6450660824775696), ('12871', 0.6447880864143372), ('17176', 0.6443230509757996), ('8860', 0.6411486864089966), ('9258', 0.6396166682243347), ('5636', 0.6394439935684204), ('15846', 0.6379180550575256), ('15891', 0.6378426551818848), ('15253', 0.6375370621681213), ('594', 0.6370989084243774)]
# 5160-->[('9736', 0.9998531341552734), ('4067', 0.9998525381088257), ('496', 0.9998511075973511), ('10099', 0.9998486638069153), ('10226', 0.9998483061790466), ('15266', 0.9998416900634766), ('15408', 0.9998409152030945), ('13649', 0.9998403787612915), ('5925', 0.9998400211334229), ('17950', 0.9998394846916199)]