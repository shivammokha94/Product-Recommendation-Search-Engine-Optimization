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
from nltk import tokenize
sw = stopwords.words('english')
import json, re, nltk, string
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from fuzzywuzzy import process
import networkx as nx
import matplotlib.pyplot as plt
lemma = nltk.wordnet.WordNetLemmatizer()

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
    
    
def count_vec(x, n= 30000):
    
    tf_vectorizer = CountVectorizer(max_features = n, stop_words = sw)
    tf = tf_vectorizer.fit_transform([x])
    tf_feature_names = tf_vectorizer.get_feature_names()
    tf_feature_names = [x for x in tf_feature_names if len(x) >= 3]
    return tf_feature_names


##########################################################################################################



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







##########################################################################################################
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
amazon_prodic = dict(amazon_db.groupby('brand')['titles'].apply(list))
#amazon_db = pd.DataFrame(amazon_db.merge(bb_az, how = 'inner', on = 'brand'))
#bbner = pd.DataFrame(pd.read_csv('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/bb_ner_ds.tsv', sep = '\t'))


for k, v in amazon_prodic.items():
    amazon_prodic[k] = sum(v, [])
    
with open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/amazon_prod.json', 'w') as fp:
    json.dump(amazon_prodic, fp, default=set_default)   
    
ama_net = nx.Graph(amazon_prodic)
nx.write_gml(ama_net, "amazon_prod.gml")

ama_net.nodes()
ama_net.edges("apple")
ama_net.neighbors("apple")
    
##########################################################################################################

userdf = preprocess.start_preprocess()
userdf = pd.DataFrame(list(userdf.items()), columns = ['i_id', 'text'])
userdf['text'] = userdf['text'].apply(lambda x: count_vec(x, 30))
userdf_prodic = dict(userdf.groupby('i_id')['text'].apply(list))

for k, v in userdf_prodic.items():
    userdf_prodic[k] = sum(v, [])
    
with open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/user_key.json', 'w') as fp:
    json.dump(userdf_prodic, fp, default=set_default)

user_net = nx.Graph(userdf_prodic)
nx.write_gml(user_net, "user_network.gml")


userdf
userdf_prodic





##########################################################################################################

str2Match = "life, mom, motherhood, organized, school, spring"
strOptions = ["Cream Spring: Clothing"]
Ratios = process.extract(str2Match,strOptions)
print(Ratios)

# You can also select the string with the highest matching percentage
highest = process.extractOne(str2Match,strOptions)
print(highest)
  



