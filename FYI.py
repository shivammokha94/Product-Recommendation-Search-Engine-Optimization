#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:09:36 2019

@author: smokha
"""


#!/usr/bin/python python3.6

# Import necessary packages
import json
import os
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk

nltk.download('wordnet')                                                        #Added the wordnet dictionary

#from sklearn.decomposition import LatentDirichletAllocation

# Load the file location
current_file_location = os.path.abspath(os.path.dirname('/'))
tweets_file_location = os.path.join(current_file_location+"Users/smokha/tweets/")             #Make Directory changes here

def get_keywords_frm_file(file_loc):
    try:
        
        # Convert to dataframe
        tweets = pd.read_csv(influencer_file_location, sep=",\'", header=None, engine='python')          #Added python engine
        tweets.columns = ["id", "tweet"]
        
	# Replace url links & mentions in the tweet column
        tweets['tweet'] = tweets['tweet'].str.replace('http\S+|www.\S+|htt\S+|\d+', '', case=False)
        tweets['tweet'] = tweets['tweet'].str.replace('@\S+', '', case=False)
        
	# Create lemmatizer
       	lemma = nltk.wordnet.WordNetLemmatizer()
        tweets['tweet'] = tweets['tweet'].apply(lambda x: [lemma.lemmatize(y) for y in x.split()]) 
        tweets['tweet'] = tweets['tweet'].apply(" ".join)
        
	# Define custom stopwords, number of topics and features
       	my_stop_words = text.ENGLISH_STOP_WORDS.union(["rt", "new", "post", "thank", "amp", "thank", "today", "yes", "just", "day", "oh", "ve", "like", "got", "xo", "need", "awh", "ll", "haha"])
	# number_of_topics = 5
       	number_of_features = 25
        
	# Define vectorizer from scikit learn package
       	tf_vectorizer = CountVectorizer(max_features = number_of_features, stop_words = my_stop_words)
        tf = tf_vectorizer.fit_transform(tweets["tweet"])
        tf_feature_names = tf_vectorizer.get_feature_names()
        
	# Remove words of length 2
       	tf_feature_names = [x for x in tf_feature_names if len(x) >= 3]
        print (tf_feature_names)
        return tf_feature_names
    except:
        return []




overall_tweets_files = os.listdir(tweets_file_location)
overall_json = {}
counter = len(overall_tweets_files)

for each in overall_tweets_files:
    influencer_id = each.split(".")[0]                                          #Remove txt extension
    influencer_file_location = os.path.join(tweets_file_location+each)          #Fetching the respective file name
    overall_json[influencer_id] = get_keywords_frm_file(influencer_file_location)
    counter -= 1
    print (counter, "Influencers remaining")

with open("keywords.json", "w") as fp:
    json.dump(overall_json, fp)



overall_json