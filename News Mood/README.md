

```python
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
now = datetime.datetime.now()
import seaborn as sns
sns.set(color_codes=True)
from config import consumer_key,consumer_secret,access_token,access_token_secret
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Target User Accounts
target_user = ["@BBCWorld", "@CBS", "@CNN","@FoxNews","@nytimes"]


```


```python
# Loop through each user
compound_list=[]
positive_list = []
negative_list = []
neutral_list = []
dates=[]
account=[]
texts=[]
for user in target_user:
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user)

        # Loop through all tweets
        for tweet in public_tweets:
            account.append(user)
            texts.append(tweet['text'])
            dates.append(tweet['created_at'])

            # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]


            # Add each value to the appropriate array
            compound_list.append(compound)
            positive_list.append(pos)
            neutral_list.append(neu)
            negative_list.append(neg)

```


```python
e={'Compound Score':compound_list,'Positive Score':positive_list,'Negative Score':negative_list,'Neutral Score':neutral_list,'Time of Tweet':dates,'Author of Tweet':account,'Text':texts}
f=pd.DataFrame(e)
f.to_csv('output.csv', sep=',')
f.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Author of Tweet</th>
      <th>Compound Score</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
      <th>Positive Score</th>
      <th>Text</th>
      <th>Time of Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Can the ad industry persuade us to eat more ve...</td>
      <td>Wed Mar 28 00:41:30 +0000 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCWorld</td>
      <td>0.2732</td>
      <td>0.0</td>
      <td>0.792</td>
      <td>0.208</td>
      <td>5 pop songs you didn't know were about God htt...</td>
      <td>Wed Mar 28 00:33:33 +0000 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCWorld</td>
      <td>0.4404</td>
      <td>0.0</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>Kim Jong-un's Beijing visit is considered a si...</td>
      <td>Wed Mar 28 00:16:30 +0000 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCWorld</td>
      <td>0.5859</td>
      <td>0.0</td>
      <td>0.817</td>
      <td>0.183</td>
      <td>@BBCBreaking North Korea's leader Kim Jong-un ...</td>
      <td>Wed Mar 28 00:04:15 +0000 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>China confirms Kim Jong-un visit https://t.co/...</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
    </tr>
  </tbody>
</table>
</div>




```python
xval=[i for i in range(100)]
for i in target_user:
    plt.scatter(xval,f.groupby('Author of Tweet').get_group(i)['Compound Score'])
plt.legend(target_user,bbox_to_anchor=(.3, 0, 1., 1.02),title='Media Sources')
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.title('Sentiment Analysis of Media Tweets '+ now.strftime("%Y-%m-%d"))
plt.show()
```


![png](output_4_0.png)



```python
#Averages
averages=f.groupby('Author of Tweet')['Compound Score'].mean()
strings=[]
for i in averages:
    strings.append('{0:.2f}'.format(i))
abovezero=[]
for i in averages:
    if i>0:
        abovezero.append('g')
    else:
        abovezero.append('r')
```


```python
#Tumor Change Graph
x_axis = np.arange(len(averages))
plt.bar(x_axis,averages, color=abovezero, align="edge",alpha=.5)
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, target_user)
plt.xlabel('Media Networks')
plt.ylabel('Tweet Polarity')
plt.title('Overall Media Sentiment based on Twitter '+ now.strftime("%Y-%m-%d"))
for i in range(len(averages)):
    if averages[i]>0:
        plt.text(x=x_axis[i]+.2,y=averages[i]+.01,s=strings[i])
    else:
        plt.text(x=x_axis[i]+.15,y=averages[i]-.03,s=strings[i])
plt.show()
```


![png](output_6_0.png)

