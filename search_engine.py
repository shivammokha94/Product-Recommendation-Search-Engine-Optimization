#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:09:57 2019

@author: smokha
"""
import pandas as pd
from fuzzywuzzy import fuzz
import json, operator
import spacy
import os

curr_path = os.getcwd()

##########################################################################################################
#Spacy model load for POS tagging
nlp = spacy.load("en_core_web_sm")


#JSON file reader
import spacy
nlp = spacy.load('en')


def json_reader(path):
    with open(path) as json_file:  
        a = json.load(json_file)      
    for k, v in a.items():
        a[k] = sum(v, [])  
    df = pd.DataFrame(list(a.items()), columns = ['brand', 'titles'])
    return df

##########################################################################################################
#Search exact product word in dataframe column and return boolean
def string_search(word, x):
    if word in x.split():
        return True
    else: 
        return False


#Part of Speech tagger to remove adjectives, verbs and extra verbage
def pos_tagger(data):
    doc = nlp(data)
    tagged_sent = [(token.text, token.tag_) for token in doc]
    tagged_sent = [word for word,tag in tagged_sent if tag != 'DT' and tag != 'JJ' and tag != 'JJS' and tag != 'RB']
    return tagged_sent
    
    

##########################################################################################################
#Read Product dictionary (for string version)

amaz = '/amazon_prod.json'    

amazon_st = json_reader(os.path.join(curr_path+amaz))    #CHANGE DIR HERE
amazon_st['titles'] = amazon_st['titles'].apply(" ".join)   #Convert titles list to string 
amazon_st['titles_tag'] = amazon_st['titles'].apply(lambda x: pos_tagger(x))  #POS tagger function call
cols = ['brand', 'titles']
amazon_st['titles'] = amazon_st[cols].apply(lambda x: ' '.join(x.astype(str)), axis=1) #Adding brands to titles
amazon_st['titles'] #Data check


user = '/user_key.json'

#Read User dictionary (for string version)
with open(os.path.join(curr_path+user)) as json_file:  #CHANGE DIR HERE
    u = json.load(json_file) 

user_st = pd.DataFrame(list(u.items()), columns = ['id', 'key'])
user_st['key'] = user_st['key'].apply(" ".join)     #Convert titles list to string  
user_st = user_st[user_st['key'] != 'nan']   #Remove any null values
user_st['key'] = user_st['key'].apply(lambda x: pos_tagger(x)) #POS tagger function call




#Mimic Search Engine #
##########################################################################################################
######## RUN FROM HERE TILL ---

se_st = 'apple iphone' #Search Phrase
#Examples: 'apple iphone', 'samsung note', 'dell laptop'


#Get token set ratio from fuzzywuzzy string comparision
amazon_st['tset_ratio'] = amazon_st['titles'].apply(lambda x: fuzz.token_set_ratio(x,se_st))


amazon_st.sort_values(by=['tset_ratio'], ascending = False)    #Data check



amazon_st_temp = amazon_st[amazon_st['tset_ratio'] > 80]   #Set threshold for product list 
amazon_st_temp['titles'] = amazon_st_temp['titles'].apply(lambda x:x.split())  #Convert string to list within DataFrame
prod_ls = amazon_st_temp["titles"].tolist() #Convert to product list
prod_ls = sum(prod_ls, []) #Remove any extra list so added


str(prod_ls)   #Check product list here
prod_ls = list(set(prod_ls))   #Set to remove duplicates



user_st_temp = pd.DataFrame   #Temp DataFrame
els = []  #Empty list


########    Approach 1

#For loop to check for each word in product list
for word in prod_ls:
    user_st['partial_ratio'] = user_st['key'].apply(lambda x: fuzz.partial_ratio(x, word)) #Get fuzzywuzzy partial ratio
    temp = user_st[user_st['partial_ratio'] == 100] #Get values only for perfect match
    if not temp.empty:
        els.append(temp)  #Append DF to list


########    Approach 2
#for word in prod_ls:
#    user_st['string_bool'] = user_st['key'].apply(lambda x: string_search(word, x))
#    temp = user_st[user_st['string_bool'] == True]
#    if not temp.empty:
#        els.append(temp)


user_st_temp = pd.concat(els, ignore_index=True)  #Convert list of Dataframes to single DataFrame

user_ls = user_st_temp["id"].tolist()   #Convert to list to obtain a list of all influencer ID's

final_inf_dic = {}  #Empty Dictionary
for i in user_ls:  #For loop to obtain influencers and the count of how many words they have mentioned from the product list
    final_inf_dic[i] = final_inf_dic.get(i, 0) + 1    

#Sort by descending order
final_inf_dic = sorted(final_inf_dic.items(), key=operator.itemgetter(1), reverse = True)
#Check here for final values of influencer

#user_st_temp['id'].value_counts() #Shortcut to value count instead of dictionary




################ Saving string DF of amazon and user dic
#amazon_st.to_csv('fuzz.csv')
#user_st.to_csv('user_st.csv')



######## RUN TILL HERE ---


##########################################################################################################



#
#
##Extra 
#Str1 = "The sup court case of Nixon vs The United States"
#Str2 = "supreme"
#Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
#Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
#Token_Sort_Ratio = fuzz.token_sort_ratio(Str1,Str2)
#Token_Set_Ratio = fuzz.token_set_ratio(Str1,Str2)
#
#
#s = 'bblogger bbloggers bbloggersca check enter follow fotd get igbeauty lip lipstick love one pack palette'
#
##
#
#Str1 = "The sup court case of Nixon vs The United States"
#Str2 = "supreme"
#Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
#Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
#Token_Sort_Ratio = fuzz.token_sort_ratio(Str1,Str2)
#Token_Set_Ratio = fuzz.token_set_ratio(Str1,Str2)
#
#
