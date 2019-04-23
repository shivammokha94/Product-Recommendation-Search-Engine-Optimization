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
import json, nltk
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
lemma = nltk.wordnet.WordNetLemmatizer()
import os

curr_path = os.getcwd()
##########################################################################################################
#For conversion to dictionary

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError
    
 
    
##########################################################################################################
#CountVectorizer Function

def count_vec(x, n= 30000):
    
    tf_vectorizer = CountVectorizer(max_features = n, stop_words = sw)
    tf = tf_vectorizer.fit_transform([x])
    tf_feature_names = tf_vectorizer.get_feature_names()
    tf_feature_names = [x for x in tf_feature_names if len(x) >= 3]
    return tf_feature_names


##########################################################################################################
#Changing and cleaning CSV for better readablity

brands_org = '/brands.txt'
titles_org = '/titles.txt'
brands_path = '/brands_1.txt'
titles_path = '/titles_1.txt'

titles = open(os.path.join(curr_path+titles_org), 'r+', encoding="latin-1")
titles = pd.DataFrame(titles.readlines())
titles = titles.apply(lambda x: x.str.replace('|',''))
titles = titles.apply(lambda x: x.str.replace(' ','|',1))
titles.head()
titles.info()
np.savetxt(os.path.join(curr_path+titles_path), titles.values, fmt='%s', newline='')

brand = open(os.path.join(curr_path+brands_org), 'r+', encoding="latin-1")
brand = pd.DataFrame(brand.readlines())
brand = brand.apply(lambda x: x.str.replace('|',''))
brand = brand.apply(lambda x: x.str.replace(' ','|',1))
brand.head()
brand.info()
np.savetxt(os.path.join(curr_path+brands_path), brand.values, fmt='%s', newline='')


##########################################################################################################
#Merging 3 Product Databases

bb_diepie = '/diepiedata.csv'
dd_turk = '/bb_ecomm.csv'

#Database 1 - AMAZON DB
brands_pd = pd.DataFrame(pd.read_csv(os.path.join(curr_path+brands_path), sep = "|", header = None))
brands_pd.columns = ['asin', 'brand']

titles_pd = pd.DataFrame(pd.read_csv(os.path.join(curr_path+titles_path), sep = "|", header = None))
titles_pd.columns = ['asin', 'titles']
amazon_db = pd.DataFrame(titles_pd.merge(brands_pd, how = 'inner', on = 'asin'))
amazon_db = amazon_db[amazon_db['brand'] != 'NaN']


#Database 2 - BestBuy e-commerce electronic DB
bb_az = pd.DataFrame(pd.read_csv(os.path.join(curr_path+bb_diepie), sep = ","))
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

#Database 3 - DataTurks DB
bb_ecomm = pd.DataFrame(pd.read_csv(os.path.join(curr_path+dd_turk), sep = ",", encoding = 'latin-1', ))
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




amaz = '/amazon_prod.json'

#Conversion to Dictionary
for k, v in amazon_prodic.items():
    amazon_prodic[k] = sum(v, [])
    
with open(os.path.join(curr_path+amaz), 'w') as fp:
    json.dump(amazon_prodic, fp, default=set_default)   
    
    
#Conversion to Graph
ama_net = nx.Graph(amazon_prodic)
nx.write_gml(ama_net, "amazon_prod.gml")
ama_net.nodes()
ama_net.edges("apple")
ama_net.neighbors("apple")
    
##########################################################################################################
#Function call to preprocess.py and reading users
userdf = preprocess.start_preprocess()
userdf = pd.DataFrame(list(userdf.items()), columns = ['i_id', 'text'])
userdf['text'] = userdf['text'].apply(lambda x: count_vec(x, 30))
userdf_prodic = dict(userdf.groupby('i_id')['text'].apply(list))


prod = '/user_key.json'

#Conversion to Dictionary
for k, v in userdf_prodic.items():
    userdf_prodic[k] = sum(v, [])
    
with open(os.path.join(curr_path+prod), 'w') as fp:
    json.dump(userdf_prodic, fp, default=set_default)

#Conversion to Graph
user_net = nx.Graph(userdf_prodic)
nx.write_gml(user_net, "user_network.gml")

#Read Check
userdf
userdf_prodic
##########################################################################################################




