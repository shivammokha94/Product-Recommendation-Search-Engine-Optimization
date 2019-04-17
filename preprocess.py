#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: smokha
"""

import os, string, re, requests, nltk, json
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup




###############################################################################
#Initial preprocess conversion

dest_file_loc=''
curr_path = os.getcwd()
half_path = '/tweets_updated/'
if (os.path.exists(curr_path+half_path) == False):
    os.mkdir(curr_path+half_path)
dest_file_loc = os.path.join(curr_path+half_path)


def get_twitter_updated(tweet_file, each):
    
    
    tweets = open(tweet_file, 'r+')
    tweets = pd.DataFrame(tweets.readlines())
    tweets = tweets.dropna(how = 'any')
    dest = os.path.join(dest_file_loc+each)
    tweets = tweets.apply(lambda x: x.str.replace('|',''))
    tweets = tweets.apply(lambda x: x.str.replace(',','|',1))
    tweets.columns = ['id_data']
    tweets.fillna(np.NaN)
    tweets['id'], tweets['data'] = tweets['id_data'].str.split('|', 1).str
    tweets = tweets.drop(['id', 'id_data'], axis = 1)
    np.savetxt(dest, tweets.values, fmt='%s', newline='')


###############################################################################
#Slang soup

resp = requests.get('http://www.netlingo.com/acronyms.php', verify =False)
soup = BeautifulSoup(resp.text, "html.parser")
slangdict= {}
key=""
value=""
for div in soup.findAll('div', attrs={'class':'list_box3'}):
    for li in div.findAll('li'):
        for a in li.findAll('a'):
            key =a.text
        value = li.text.split(key)[1]
        slangdict[key]=value

slang_path = '/slang.json'        
with open(os.path.join(curr_path+slang_path), 'w') as find:
    json.dump(slangdict,find,indent=2)
    

###############################################################################

lemma = nltk.wordnet.WordNetLemmatizer()
nltk.download('wordnet')  
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
   
splchar = re.compile('[^A-Za-z0-9]+')
punc = string.punctuation
dig = string.digits
pundig = re.compile(r'[%s%s]'%(dig,punc)) 
sw = stopwords.words('english')
sw.extend(["rt", "new", "post", "thank", "amp", "thank", "today", "yes", "just", "day", "oh", "ve", "like", "got", "xo", "need", "awh", "ll", "haha"])
am = re.compile(r'amazon com')

#Slang clean
sl = slangdict.keys()
sl = ' '.join(sl)
sl = sl.lower()
sl = splchar.sub(' ', sl)
sl = pundig.sub(' ', sl)
sl = re.sub(r'\b\w{1,3}\b', '', sl)
slang = re.compile(r'%s'%sl)

###############################################################################

def cln(col):
    
    col = col.lower()
    col = emoji_pattern.sub(' ', col)
    col = splchar.sub(' ', col)
    col = pundig.sub(' ', col)
    col = slang.sub(' ', col)
    col = am.sub(' ', col)
    senlist = col.split()
    textclean = ' '.join([w for w in senlist if w.lower() not in sw])
    return textclean
    
###############################################################################
    
def get_insta_preprocessed(insta):

    #Instagram file preprocessing

    cols = ['TIPP_POST1_TEXT', 'TIPP_POST2_TEXT', 'TIPP_POST3_TEXT', 'TIPP_POST4_TEXT', 'TIPP_POST5_TEXT','TIPP_POST6_TEXT', 'TIPP_POST7_TEXT', 'TIPP_POST8_TEXT', 'TIPP_POST9_TEXT', 'TIPP_POST10_TEXT', 'TIPP_POST11_TEXT', 'TIPP_POST12_TEXT' ]
    coldrop = ['TIPP_TOTAL_POST', 'TIPP_POST1_LIKES', 'TIPP_POST2_LIKES', 'TIPP_POST3_LIKES', 'TIPP_POST4_LIKES', 'TIPP_POST5_LIKES', 'TIPP_POST6_LIKES', 'TIPP_POST7_LIKES', 'TIPP_POST8_LIKES', 'TIPP_POST9_LIKES', 'TIPP_POST10_LIKES', 'TIPP_POST11_LIKES', 'TIPP_POST12_LIKES', 'TIPP_POST1_HASHTAGS', 'TIPP_POST2_HASHTAGS', 'TIPP_POST3_HASHTAGS', 'TIPP_POST4_HASHTAGS', 'TIPP_POST5_HASHTAGS', 'TIPP_POST6_HASHTAGS', 'TIPP_POST7_HASHTAGS', 'TIPP_POST8_HASHTAGS', 'TIPP_POST9_HASHTAGS', 'TIPP_POST10_HASHTAGS', 'TIPP_POST11_HASHTAGS', 'TIPP_POST12_HASHTAGS', 'TIPP_POST1_COMMENTS', 'TIPP_POST2_COMMENTS', 'TIPP_POST3_COMMENTS', 'TIPP_POST4_COMMENTS', 'TIPP_POST5_COMMENTS', 'TIPP_POST6_COMMENTS', 'TIPP_POST7_COMMENTS', 'TIPP_POST8_COMMENTS', 'TIPP_POST9_COMMENTS', 'TIPP_POST10_COMMENTS', 'TIPP_POST11_COMMENTS', 'TIPP_POST12_COMMENTS', 'TIPP_TS', 'TIPP_UPDATED', 'TIPP_FOLLOWERS', 'TIPP_FOLLOWING', 'TIPP_ID', 'Unnamed: 56', 'Unnamed: 57', 'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63', 'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67', 'Unnamed: 68', 'Unnamed: 69', 'Unnamed: 70', 'Unnamed: 71']

    insta.isna().sum()                                           #Check null values with respect to rows
    insta = insta.drop(coldrop, axis = 1)
    insta = insta.dropna(how = 'all')                            #Drop rows having all values as null
    insta = insta.drop_duplicates(subset='TIPP_URL')             #Drop duplicated instagram URL
    dup = insta[insta.duplicated()]                              #Check duplicates with all rows
    insta = insta.rename(columns = {'TIPP_FU_ID':'id'})
    insta['id'] = insta['id'].astype(int)                        #Change column datatype to int
    
    insta[cols] = insta[cols].apply(lambda x:x.str.replace('@\S+', '', case=False))
    insta[cols] = insta[cols].apply(lambda x:x.str.replace('rt|http\S+|www.\S+|htt\S+|\d+', '', case=False))

    insta = insta.fillna(value = 'thistextismissingdata')
    for each in cols:
        insta[each] = insta[each].apply(lambda x:cln(x)) 
        insta[each] = insta[each].apply(lambda x: [lemma.lemmatize(y, pos = 'v') for y in x.split()])
        insta[each] = insta[each].apply(" ".join)
    insta = insta.replace('thistextismissingdata', np.NaN)
    
    return insta


###############################################################################
    
def get_data(file):
    
    tweets = pd.DataFrame(pd.read_csv(file, header=None, engine='c', sep='\n', encoding='latin1'))          #Added python engine
    tweets.columns = ['tweet']
    tweets = tweets.dropna(how = 'any')
    
    tweets['tweet'] = tweets['tweet'].str.replace('@\S+', '', case=False)
    tweets['tweet'] = tweets['tweet'].str.replace('rt|http\S+|www.\S+|htt\S+|\d+', '', case=False)
    
    tweets['tweet'] = tweets['tweet'].apply(lambda x: cln(x))
    tweets['tweet'] = tweets['tweet'].apply(lambda x: [lemma.lemmatize(y, pos='v') for y in x.split()]) 
    tweets['tweet'] = tweets['tweet'].apply(" ".join)
    t = ' '.join(tweets['tweet'])
    
    return t
      
def get_twitter_preprocessed(file_list):

    #Tweets data preprocessing
    tweet = pd.DataFrame()
    counter = len(file_list)

    for i in file_list: 
        inf_id = i.split('.')[0]
        inf_file_loc = os.path.join(dest_file_loc+i)
        tweet = tweet.append({'id':inf_id, 'tweetdata':get_data(inf_file_loc)}, ignore_index=True)
        counter -= 1
        print (counter, "Influencers remaining")
        
    return tweet

###############################################################################

def start_preprocess():
    
#Instagram preprocessing
    insta_half_path = '/insta_posts.csv'
    insta = pd.DataFrame(pd.read_csv(os.path.join(curr_path+insta_half_path)))
    print('Starting Instagram preprocessing')
    insta = get_insta_preprocessed(insta)
    print('End of Instagram preprocessing')



#Twitter preprocessing
    twitter_half_path = '/tweets/'
    tweets_file_loc = os.path.join(curr_path+twitter_half_path)
    print('Starting Twitter initial preprocessing')
    file_list = os.listdir(tweets_file_loc)


    counter = len(file_list)
    for each in file_list:
        influencer_id = each.split(".")[0]                                          
        influencer_file_location = os.path.join(tweets_file_loc+each)       
        get_twitter_updated(influencer_file_location, each)
        counter -= 1
        print (counter, "Influencers remaining")
    print('End of Twitter initial preprocessing')

    print('Starting Twitter main preprocessing')
    twitter = get_twitter_preprocessed(file_list)
    print('End of Twitter main preprocessing')


    twitter['id'] = twitter['id'].apply(lambda x: int(x or 0))
    twitter = twitter[twitter['id'] != 0]


#Join instagram & twitter data
    instwi = pd.DataFrame(insta.merge(twitter, how = 'left', on = 'id'))

    instwi.to_csv('comb.csv')
###############################################################################

    instwi_comb = instwi

    col_f = ['TIPP_POST1_TEXT', 'TIPP_POST2_TEXT', 'TIPP_POST3_TEXT', 'TIPP_POST4_TEXT', 'TIPP_POST5_TEXT','TIPP_POST6_TEXT', 'TIPP_POST7_TEXT', 'TIPP_POST8_TEXT', 'TIPP_POST9_TEXT', 'TIPP_POST10_TEXT', 'TIPP_POST11_TEXT', 'TIPP_POST12_TEXT', 'tweetdata']

    instwi_comb['Data'] = instwi[col_f].apply(lambda x: ' '.join(x.astype(str)), axis=1)


    instwi_comb = instwi_comb.drop(col_f, axis=1)
    instwi_comb = instwi_comb.drop('TIPP_URL', axis = 1)

    instwi_comb = dict(zip(instwi_comb['id'], instwi_comb['Data']))
    
    return instwi_comb

###############################################################################

    final_path= '/tweets_corpus_stop_words/'
    if(os.path.exists(os.path.join(curr_path+final_path)) == False):
        os.mkdir(os.path.join(curr_path+final_path))

    corp_file_loc = os.path.join(curr_path+final_path)    

    c = len(instwi_comb)

    for key, values in instwi_comb.items():
        dest_corp = os.path.join(corp_file_loc+str(key))
        np.savetxt(dest_corp, [values], fmt='%s', newline='')
        c = c - 1
        print('Corpus remaining:', c)
    

