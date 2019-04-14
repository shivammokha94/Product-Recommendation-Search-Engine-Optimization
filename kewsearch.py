#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:21:34 2019

@author: smokha
"""

import pandas as pd
import preprocess
import numpy as np
from nltk.corpus import stopwords
sw = stopwords.words('english')
import json, re, nltk, string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from fuzzywuzzy import process
import networkx as nx
import matplotlib.pyplot as plt
lemma = nltk.wordnet.WordNetLemmatizer()


def count_vec(x, n= 30000):
    
    tf_vectorizer = CountVectorizer(max_features = n, stop_words = sw)
    tf = tf_vectorizer.fit_transform([x])
    tf_feature_names = tf_vectorizer.get_feature_names()
    tf_feature_names = [x for x in tf_feature_names if len(x) >= 3]
    return tf_feature_names


##########################################################################################################
#
#


titles = pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/titles.txt', delimiter = " ",header = None)

titles = open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/titles.txt', 'r+', encoding="latin-1")
titles = pd.DataFrame(titles.readlines())
titles = titles.apply(lambda x: x.str.replace('|',''))
titles = titles.apply(lambda x: x.str.replace(' ','|',1))
titles.head()
titles.info()
np.savetxt('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/titles_1.txt', titles.values, fmt='%s', newline='')

brand = open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/brands.txt', 'r+', encoding="latin-1")
brand = pd.DataFrame(brand.readlines())
brand = brand.apply(lambda x: x.str.replace('|',''))
brand = brand.apply(lambda x: x.str.replace(' ','|',1))
brand.head()
brand.info()
np.savetxt('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/brands_1.txt', brand.values, fmt='%s', newline='')



brands_pd = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/brands_1.txt', sep = "|", header = None))
brands_pd.columns = ['asin', 'brand']
titles_pd = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/Test_Folder_FYI/titles_1.txt', sep = "|", header = None))
titles_pd.columns = ['asin', 'titles']
amazon_db = pd.DataFrame(titles_pd.merge(brands_pd, how = 'inner', on = 'asin'))
amazon_db = amazon_db[amazon_db['brand'] != 'NaN']

bb_az = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/diepiedata.csv', sep = ","))

amazon_db = amazon_db.append(bb_az, ignore_index = True)
amazon_db = amazon_db.drop(['asin'], axis = 1)
amazon_db['titles'] = amazon_db['titles'].apply(lambda x: preprocess.cln(x))
amazon_db['titles'] = amazon_db['titles'].apply(lambda x: [lemma.lemmatize(y, pos = 'v') for y in x.split()])
amazon_db['titles'] = amazon_db['titles'].apply(" ".join)
amazon_prodic = dict(amazon_db.groupby('brand')['titles'].apply(list))
amazon_db = pd.DataFrame(list(amazon_prodic.items()), columns = ['brand', 'titles'])
amazon_db['titles'] = amazon_db['titles'].apply(" ".join)
amazon_db['titles'] = amazon_db['titles'].apply(lambda x: preprocess.cln(x))
amazon_db['brand'] = amazon_db['brand'].str.lower()
bb_ecomm = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/bb_ecomm.csv', sep = ",", encoding = 'latin-1', ))
amazon_db = amazon_db.append(bb_ecomm, ignore_index = True)
amazon_db['titles'] = amazon_db['titles'].apply(lambda x: [lemma.lemmatize(y, pos = 'v') for y in x.split()])
amazon_db['titles'] = amazon_db['titles'].apply(" ".join)
amazon_prodic = dict(amazon_db.groupby('brand')['titles'].apply(list))
amazon_db = pd.DataFrame(list(amazon_prodic.items()), columns = ['brand', 'titles'])
amazon_db['titles'] = amazon_db['titles'].apply(" ".join)
amazon_db['titles'] = amazon_db['titles'].apply(lambda x: count_vec(x))
#amazon_db = pd.DataFrame(amazon_db.merge(bb_az, how = 'inner', on = 'brand'))
#bbner = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/bb_ner_ds.tsv', sep = '\t'))


amazon_db
amazon_prodic
userdf = preprocess.start_preprocess()



userdf = pd.DataFrame(list(userdf.items()), columns = ['i_id', 'text'])

userdf['text'] = userdf['text'].apply(lambda x: count_vec(x, 25))

userdf



def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
    
amazon_prodic = dict(amazon_db.groupby('brand')['titles'].apply(list))
with open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/amazon_prod.json', 'w') as fp:
    json.dump(amazon_prodic, fp, default=set_default)

userdf_prodic = dict(userdf.groupby('i_id')['text'].apply(list))
with open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/user_key.json', 'w') as fp:
    json.dump(userdf_prodic, fp, default=set_default)




str2Match = "life, mom, motherhood, organized, school, spring"
strOptions = ["Cream Spring: Clothing"]
Ratios = process.extract(str2Match,strOptions)
print(Ratios)
# You can also select the string with the highest matching percentage
highest = process.extractOne(str2Match,strOptions)
print(highest)







g = nx.DiGraph()
g.add_nodes_from(amazon_prodic.keys())


for k, v in amazon_prodic.items():
    g.add_edges_from(([(k, t) for t in v]))
    
pos=nx.spring_layout(g)

p = nx.draw(g,pos,with_labels=True, node_size=60,font_size=8)
plt.draw()
plt.show()

p.savefig("plot.png", dpi=1000)

nx.write_gml(g, "amazon_prod.gml")
