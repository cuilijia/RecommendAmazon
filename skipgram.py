from gensim.models import Word2Vec

from sklearn.decomposition import PCA
from matplotlib import pyplot

import requests
import json

def getItemsInfo(id):

    itemIndex = int(id)
    params = {"index": itemIndex}
    r = requests.get('http://34.80.151.46:10086/getItemInfo', params=params)
    iteminfo=r.json()
    print(iteminfo['title'])
    request_download(str(id), iteminfo['imUrl'])
    return iteminfo['title']


def performance(input, output):
    a = open("data/Sports_and_Outdoors_Index.json", "r")
    s = json.load(a)
    list1 = []
    for i in input:
        itemIndex = int(i)
        params = {"index": itemIndex}
        r = requests.get('http://34.80.151.46:10086/getAllRelated', params=params)
        for j in r.json():
            list1.append(j)
    list2 = []
    # print(list1)
    for k in output:
        list2.append(s[int(k[0])]["asin"])
    # print(list2)
    list3 = list(set(list1).intersection(set(list2)))
    return len(list3) / len(list2)


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

    # print(str(productidlist)+"-->"+str(model.wv.most_similar(productidlist)))
    return productidlist,model.wv.most_similar(productidlist)

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

def countpre():
    with open('data/data4.json','r') as load_f:
        load_dict = json.load(load_f)
    print(len(load_dict[0]))

    sentences = []
    for i in range(len(load_dict[0])):
        # print(load_dict[0][str(i)][:9])
        # sentences.append(load_dict[0][str(i)][:9])
        data1=load_dict[0][str(i)][:9]
        input, output = test(data1)
        count = performance(input, output)
        # print(count)
        print(i)
        sentences.append(count)

        if (i%100)==0:
            print(max(sentences))

def request_download(id,url):
    import requests
    r = requests.get(url)
    with open('img/img'+id+'.png', 'wb') as f:
        f.write(r.content)

# getItemsInfo(11550)
# drawpic(0)
# 输入是画的size，为0是全部输出

# train("data/data1.json",Sg=0,Size=250,Window=5,Min_count=1,Workers=4,Iter=10)
# for i in range(2,11):
#     addtrain("data/data"+str(i)+".json")

def seachinfolist(input,output):
    outputx = []
    for o in output:
        outputx.append(o[0])

    for it in input:
        getItemsInfo(it)

    print('output:')

    for it in outputx:
        getItemsInfo(it)


input,output =test(["10929", "10556", "13271", "13359", "17527", "154", "8782", "12977"])
print(input)
print(output)
seachinfolist(input,output)

# count=performance(input, output)
# print(count)
