# Product-Recommendation-Search-Engine-Optimization
Search engine optimization using Python. This is also a product recommendation project based on Natural Language Processing.


The code, as attached without data, is my graduate capstone project based on information retrieval and natural language processing. 

The first python file, "preprocess.py", gets the social media data from twitter and instagram (for about 7000 users) data sources stored at local and preprocesses them in terms of removing slangs, emoticons, stop/stem words and special characters. 

The second python file, "kewsearch.py",feeds the data into a count vectorizer to extract keywords from the text for each user and then converts them to dictionaries where the key is the userID and the value is the corresponding keywords extracted. This python file also gets the product data (in excess of 10 GB) from 3 different open source datasets and performs count vectorizer and with it's conversion to a product dictionary. The dictionary has the product brand has key and all the products related to the product brand as values. 

The third python file, "search_engine.py" as a custom search engine based on fuzzy string matching and finds all the users who have mentioned the product related to a particular brand. For example, if the search phrase is "Apple", the result would be fetched as all the users who have mentioned products like "iPad", "iPhone", "Mac" even though they might not have mentioned "Apple" specifically.
