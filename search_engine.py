#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:09:57 2019

@author: smokha
"""
import pandas as pd
from fuzzywuzzy import fuzz
import json, operator
from collections import Counter
from difflib import SequenceMatcher
from nltk.tag import pos_tag
import spacy
nlp = spacy.load('en')



def pos_tagger(data):
    doc = nlp(data)
    tagged_sent = [(token.text, token.tag_) for token in doc]
    tagged_sent = [word for word,tag in tagged_sent if tag != 'DT' and tag != 'JJ']
    return tagged_sent
    
def append_brand(data):   

def json_reader(path):
    with open(path) as json_file:  
        a = json.load(json_file)      
    for k, v in a.items():
        a[k] = sum(v, [])  
    df = pd.DataFrame(list(a.items()), columns = ['brand', 'titles'])
    return df


amazon = json_reader('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/amazon_prod.json')
amazon_st = json_reader('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/amazon_prod.json')
amazon_st['titles'] = amazon_st['titles'].apply(" ".join)



with open('/Users/smokha/Projects/FYI_Keyword_Heirarchy_BC/user_key.json') as json_file:  
    u = json.load(json_file) 

user = pd.DataFrame(list(u.items()), columns = ['id', 'key'])
user['key'] = user['key'].apply(" ".join)
user = user[user['key'] != 'nan']
user['key'] = user['key'].apply(lambda x:x.split())

user_st = pd.DataFrame(list(u.items()), columns = ['id', 'key'])
user_st['key'] = user_st['key'].apply(" ".join)
user_st = user_st[user_st['key'] != 'nan']


amazon_st['tag'] = amazon_st['titles'].apply(lambda x: pos_tagger(x))

##########################################################################################################
se_st = 'asus laptop'


amazon_st['tset_ratio'] = amazon_st['titles'].apply(lambda x: fuzz.token_set_ratio(x,se_st))


amazon_st.sort_values(by=['tset_ratio'], ascending = False)

amazon_st_temp = amazon_st[amazon_st['tset_ratio'] > 80]
amazon_st_temp['titles'] = amazon_st_temp['titles'].apply(lambda x:x.split())
prod_ls = amazon_st_temp["titles"].tolist()
prod_ls = sum(prod_ls, [])
str(prod_ls)
prod_ls = list(set(prod_ls))

user_st_temp = pd.DataFrame
els = []
for word in prod_ls:
    user_st['partial_ratio'] = user_st['key'].apply(lambda x: fuzz.partial_ratio(x, word))
    temp = user_st[user_st['partial_ratio'] == 100]
    if not temp.empty:
        els.append(temp)

user_st_temp = pd.concat(els, ignore_index=True)
 
user_ls = user_st_temp["id"].tolist()
user_ls
b = {}
for i in user_ls:
    b[i] = b.get(i, 0) + 1    

b = sorted(b.items(), key=operator.itemgetter(1), reverse = True)
b

for i in prod_ls:
    print(i)

amazon_st.to_csv('fuzz.csv')
##########################################################################################################

Str1 = "The sup court case of Nixon vs The United States"
Str2 = "supreme"
Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
Token_Sort_Ratio = fuzz.token_sort_ratio(Str1,Str2)
Token_Set_Ratio = fuzz.token_set_ratio(Str1,Str2)


