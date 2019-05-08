import requests

import json

a = open("Sports_and_Outdoors_Index.json", "r")
s = json.load(a)


def performance(input, output, s):
    list1 = []
    for i in input:
        itemIndex = int(i)
        params = {"index": itemIndex}
        r = requests.get('http://34.80.151.46:10086/getAllRelated', params=params)
        for j in r.json():
            list1.append(j)
    list2 = []
    print(list1)
    for k in output:
        list2.append(s[int(k[0])]["asin"])
    print(list2)
    list3 = list(set(list1).intersection(set(list2)))
    return len(list3) / len(list2)


input = ["4831", "13644", "14643"]
output = [('8738', 0.996864378452301), ('11652', 0.9961963295936584), ('10471', 0.9955419301986694),
          ('2187', 0.9954941272735596), ('10385', 0.9951783418655396), ('12243', 0.9946809411048889),
          ('7672', 0.9946743249893188), ('11059', 0.9946001172065735), ('3368', 0.994516909122467),
          ('1215', 0.9943747520446777)]
count = performance(input, output, s)

'''
api 有 
/getItemInfo 获得商品信息 这里面是最全的 包含下面所有

/getAlsoBought 获得alsobought的ASIN 部分ASIN不在 random walk 的 graph matrix里 所有只返回ASIN 不是 index

/getAlsoViewed 获得alsoviewd的ASIN

/getBoughtTogether 获得alsoboughttogether的ASIN

/getAllRelated getAlsoBought+getAlsoViewed+getBoughtTogether 无重复项

'''