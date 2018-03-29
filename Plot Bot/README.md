
# Twitter Account:
https://twitter.com/RobBotro



```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Helps title graph
import datetime
now = datetime.datetime.now()
#Format graph to seaborn
import seaborn as sns
sns.set(color_codes=True)
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
#Store List of Previously Tweeted
import pickle
with open("previousrequests.txt", "rb") as fp:   # Unpickling
    previousrequests = pickle.load(fp)
# Setup Tweepy API Authentication
from config import consumer_key,consumer_secret,access_token,access_token_secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```


```python
#Get list of requests
c=api.mentions_timeline(count=500)
requests=[]
users=[]
for tweets in c:
    try:
        x,y,z=c[c.index(tweets)]['text'].split('@')
        z='@'+z
        if z.lower() not in previousrequests and tweets['user']['screen_name']!='RobBotro':
            previousrequests.append(z)
            x,y,z=c[c.index(tweets)]['text'].split('@')
            z='@'+z
            requests.append(z)
            users.append(tweets['user']['screen_name'])
    except:
        break
```


```python
#Make graph and tweet it out
for user in requests:
    # Loop through 5 pages of tweets (total 100 tweets)
    compound_list=[]
    for x in range(1,6):
        
        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)

        # Loop through all tweets
        for tweet in public_tweets:
            compound_list.append(analyzer.polarity_scores(tweet["text"])["compound"])
    plt.scatter([i for i in range(len(compound_list))],compound_list)
    plt.plot([i for i in range(len(compound_list))],compound_list)
    plt.ylabel('Tweet Polarity')
    plt.xlabel('Tweets Ago')
    plt.xlim(103, -3)
    plt.title(f'Sentiment Analysis of Tweets of {user} ' + now.strftime("%Y-%m-%d"))    

    plt.savefig('graph.jpg')
    api.update_with_media("graph.jpg",
                      f"New Tweet Analysis: {user} Thanks @{users[requests.index(user)]}")
    plt.close()
    
```


```python
#Store list of previous requests
with open("previousrequests.txt", "wb") as fp:   #Pickling
    pickle.dump(previousrequests, fp)
```
