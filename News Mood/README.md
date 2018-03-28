

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
import calendar
b=dict((v,k) for k,v in enumerate(calendar.month_abbr))

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
f
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
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Can the ad industry persuade us to eat more ve...</td>
      <td>Wed Mar 28 00:41:30 +0000 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCWorld</td>
      <td>0.2732</td>
      <td>0.000</td>
      <td>0.792</td>
      <td>0.208</td>
      <td>5 pop songs you didn't know were about God htt...</td>
      <td>Wed Mar 28 00:33:33 +0000 2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCWorld</td>
      <td>0.4404</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>Kim Jong-un's Beijing visit is considered a si...</td>
      <td>Wed Mar 28 00:16:30 +0000 2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCWorld</td>
      <td>0.5859</td>
      <td>0.000</td>
      <td>0.817</td>
      <td>0.183</td>
      <td>@BBCBreaking North Korea's leader Kim Jong-un ...</td>
      <td>Wed Mar 28 00:04:15 +0000 2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>China confirms Kim Jong-un visit https://t.co/...</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
    </tr>
    <tr>
      <th>5</th>
      <td>@BBCWorld</td>
      <td>-0.7841</td>
      <td>0.463</td>
      <td>0.537</td>
      <td>0.000</td>
      <td>Fired Vancouver waiter case: Are the French re...</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>@BBCWorld</td>
      <td>0.5095</td>
      <td>0.000</td>
      <td>0.798</td>
      <td>0.202</td>
      <td>RT @BBCDanielS: BBC Exclusive interview with S...</td>
      <td>Tue Mar 27 21:37:00 +0000 2018</td>
    </tr>
    <tr>
      <th>7</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>This 'smoking' elephant in India is baffling e...</td>
      <td>Tue Mar 27 21:20:33 +0000 2018</td>
    </tr>
    <tr>
      <th>8</th>
      <td>@BBCWorld</td>
      <td>-0.2516</td>
      <td>0.229</td>
      <td>0.603</td>
      <td>0.168</td>
      <td>Spy poisoning: 'I would really like to know ho...</td>
      <td>Tue Mar 27 20:14:24 +0000 2018</td>
    </tr>
    <tr>
      <th>9</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Brazilian sports reporters tackle on-air gropi...</td>
      <td>Tue Mar 27 19:19:12 +0000 2018</td>
    </tr>
    <tr>
      <th>10</th>
      <td>@BBCWorld</td>
      <td>0.6705</td>
      <td>0.130</td>
      <td>0.538</td>
      <td>0.332</td>
      <td>RT @BBCNews: Owl ring-bearer attacks best man ...</td>
      <td>Tue Mar 27 18:14:45 +0000 2018</td>
    </tr>
    <tr>
      <th>11</th>
      <td>@BBCWorld</td>
      <td>-0.2500</td>
      <td>0.222</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>Nearly a million watch cat stuck up pole https...</td>
      <td>Tue Mar 27 18:13:18 +0000 2018</td>
    </tr>
    <tr>
      <th>12</th>
      <td>@BBCWorld</td>
      <td>-0.5106</td>
      <td>0.202</td>
      <td>0.798</td>
      <td>0.000</td>
      <td>RT @BBCSport: Australia coach Darren Lehmann d...</td>
      <td>Tue Mar 27 18:02:09 +0000 2018</td>
    </tr>
    <tr>
      <th>13</th>
      <td>@BBCWorld</td>
      <td>-0.4588</td>
      <td>0.143</td>
      <td>0.857</td>
      <td>0.000</td>
      <td>RT @richard_conway: Sanctions announced agains...</td>
      <td>Tue Mar 27 17:57:01 +0000 2018</td>
    </tr>
    <tr>
      <th>14</th>
      <td>@BBCWorld</td>
      <td>0.7964</td>
      <td>0.000</td>
      <td>0.664</td>
      <td>0.336</td>
      <td>I want to apologise to "kids who love cricket ...</td>
      <td>Tue Mar 27 17:51:11 +0000 2018</td>
    </tr>
    <tr>
      <th>15</th>
      <td>@BBCWorld</td>
      <td>0.2023</td>
      <td>0.000</td>
      <td>0.913</td>
      <td>0.087</td>
      <td>RT @richard_conway: Smith, Bancroft and Warner...</td>
      <td>Tue Mar 27 17:41:16 +0000 2018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>@BBCWorld</td>
      <td>-0.3400</td>
      <td>0.194</td>
      <td>0.806</td>
      <td>0.000</td>
      <td>Russia fire: Protester spoke on the phone to d...</td>
      <td>Tue Mar 27 17:40:24 +0000 2018</td>
    </tr>
    <tr>
      <th>17</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @bbctms: James Sutherland says only 3 peopl...</td>
      <td>Tue Mar 27 17:34:28 +0000 2018</td>
    </tr>
    <tr>
      <th>18</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @richard_conway: Full statement from Cricke...</td>
      <td>Tue Mar 27 17:33:54 +0000 2018</td>
    </tr>
    <tr>
      <th>19</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Australia ball-tampering: Steve Smith, David W...</td>
      <td>Tue Mar 27 17:32:03 +0000 2018</td>
    </tr>
    <tr>
      <th>20</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Can the ad industry persuade us to eat more ve...</td>
      <td>Wed Mar 28 00:41:30 +0000 2018</td>
    </tr>
    <tr>
      <th>21</th>
      <td>@BBCWorld</td>
      <td>0.2732</td>
      <td>0.000</td>
      <td>0.792</td>
      <td>0.208</td>
      <td>5 pop songs you didn't know were about God htt...</td>
      <td>Wed Mar 28 00:33:33 +0000 2018</td>
    </tr>
    <tr>
      <th>22</th>
      <td>@BBCWorld</td>
      <td>0.4404</td>
      <td>0.000</td>
      <td>0.804</td>
      <td>0.196</td>
      <td>Kim Jong-un's Beijing visit is considered a si...</td>
      <td>Wed Mar 28 00:16:30 +0000 2018</td>
    </tr>
    <tr>
      <th>23</th>
      <td>@BBCWorld</td>
      <td>0.5859</td>
      <td>0.000</td>
      <td>0.817</td>
      <td>0.183</td>
      <td>@BBCBreaking North Korea's leader Kim Jong-un ...</td>
      <td>Wed Mar 28 00:04:15 +0000 2018</td>
    </tr>
    <tr>
      <th>24</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>China confirms Kim Jong-un visit https://t.co/...</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
    </tr>
    <tr>
      <th>25</th>
      <td>@BBCWorld</td>
      <td>-0.7841</td>
      <td>0.463</td>
      <td>0.537</td>
      <td>0.000</td>
      <td>Fired Vancouver waiter case: Are the French re...</td>
      <td>Tue Mar 27 23:50:38 +0000 2018</td>
    </tr>
    <tr>
      <th>26</th>
      <td>@BBCWorld</td>
      <td>0.5095</td>
      <td>0.000</td>
      <td>0.798</td>
      <td>0.202</td>
      <td>RT @BBCDanielS: BBC Exclusive interview with S...</td>
      <td>Tue Mar 27 21:37:00 +0000 2018</td>
    </tr>
    <tr>
      <th>27</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>This 'smoking' elephant in India is baffling e...</td>
      <td>Tue Mar 27 21:20:33 +0000 2018</td>
    </tr>
    <tr>
      <th>28</th>
      <td>@BBCWorld</td>
      <td>-0.2516</td>
      <td>0.229</td>
      <td>0.603</td>
      <td>0.168</td>
      <td>Spy poisoning: 'I would really like to know ho...</td>
      <td>Tue Mar 27 20:14:24 +0000 2018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>@BBCWorld</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Brazilian sports reporters tackle on-air gropi...</td>
      <td>Tue Mar 27 19:19:12 +0000 2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>@nytimes</td>
      <td>-0.7096</td>
      <td>0.228</td>
      <td>0.772</td>
      <td>0.000</td>
      <td>RT @jdelreal: About two dozen people are gathe...</td>
      <td>Tue Mar 27 22:32:02 +0000 2018</td>
    </tr>
    <tr>
      <th>471</th>
      <td>@nytimes</td>
      <td>-0.5994</td>
      <td>0.224</td>
      <td>0.776</td>
      <td>0.000</td>
      <td>Apple is introducing a new iPad, trying to bec...</td>
      <td>Tue Mar 27 22:17:02 +0000 2018</td>
    </tr>
    <tr>
      <th>472</th>
      <td>@nytimes</td>
      <td>0.0772</td>
      <td>0.000</td>
      <td>0.939</td>
      <td>0.061</td>
      <td>The 2020 census is a snapshot of America that ...</td>
      <td>Tue Mar 27 22:01:06 +0000 2018</td>
    </tr>
    <tr>
      <th>473</th>
      <td>@nytimes</td>
      <td>-0.6249</td>
      <td>0.221</td>
      <td>0.779</td>
      <td>0.000</td>
      <td>RT @cliffordlevy: Some Republicans fear the mi...</td>
      <td>Tue Mar 27 21:47:02 +0000 2018</td>
    </tr>
    <tr>
      <th>474</th>
      <td>@nytimes</td>
      <td>-0.8225</td>
      <td>0.364</td>
      <td>0.636</td>
      <td>0.000</td>
      <td>A fatal helicopter crash in New York’s East Ri...</td>
      <td>Tue Mar 27 21:32:51 +0000 2018</td>
    </tr>
    <tr>
      <th>475</th>
      <td>@nytimes</td>
      <td>-0.8860</td>
      <td>0.309</td>
      <td>0.691</td>
      <td>0.000</td>
      <td>He was wrongfully convicted of rape and murder...</td>
      <td>Tue Mar 27 21:16:03 +0000 2018</td>
    </tr>
    <tr>
      <th>476</th>
      <td>@nytimes</td>
      <td>-0.0772</td>
      <td>0.147</td>
      <td>0.734</td>
      <td>0.119</td>
      <td>Here are 11 movies you won’t want to miss http...</td>
      <td>Tue Mar 27 21:00:23 +0000 2018</td>
    </tr>
    <tr>
      <th>477</th>
      <td>@nytimes</td>
      <td>0.0772</td>
      <td>0.000</td>
      <td>0.933</td>
      <td>0.067</td>
      <td>If President Trump actually meets Kim Jong-un,...</td>
      <td>Tue Mar 27 20:45:08 +0000 2018</td>
    </tr>
    <tr>
      <th>478</th>
      <td>@nytimes</td>
      <td>-0.2732</td>
      <td>0.110</td>
      <td>0.890</td>
      <td>0.000</td>
      <td>“The Americans” has always been as much about ...</td>
      <td>Tue Mar 27 20:30:04 +0000 2018</td>
    </tr>
    <tr>
      <th>479</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Roseanne believes President Trump doesn't oppo...</td>
      <td>Tue Mar 27 20:22:03 +0000 2018</td>
    </tr>
    <tr>
      <th>480</th>
      <td>@nytimes</td>
      <td>-0.1511</td>
      <td>0.200</td>
      <td>0.667</td>
      <td>0.133</td>
      <td>RT @jdelreal: Stephon Clark’s brother is inter...</td>
      <td>Wed Mar 28 00:41:04 +0000 2018</td>
    </tr>
    <tr>
      <th>481</th>
      <td>@nytimes</td>
      <td>-0.4404</td>
      <td>0.172</td>
      <td>0.828</td>
      <td>0.000</td>
      <td>Some might find the idea of speed-solving @nyt...</td>
      <td>Wed Mar 28 00:32:04 +0000 2018</td>
    </tr>
    <tr>
      <th>482</th>
      <td>@nytimes</td>
      <td>-0.4404</td>
      <td>0.132</td>
      <td>0.868</td>
      <td>0.000</td>
      <td>At least 12 states signaled that they would su...</td>
      <td>Wed Mar 28 00:17:05 +0000 2018</td>
    </tr>
    <tr>
      <th>483</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>RT @jdelreal: Packed house today at the Sacram...</td>
      <td>Wed Mar 28 00:08:05 +0000 2018</td>
    </tr>
    <tr>
      <th>484</th>
      <td>@nytimes</td>
      <td>0.2732</td>
      <td>0.000</td>
      <td>0.896</td>
      <td>0.104</td>
      <td>The surprise discussions added another layer o...</td>
      <td>Wed Mar 28 00:03:00 +0000 2018</td>
    </tr>
    <tr>
      <th>485</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Breaking News: North Korea's leader is said to...</td>
      <td>Tue Mar 27 23:52:39 +0000 2018</td>
    </tr>
    <tr>
      <th>486</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>As you walk into a room at University of Calif...</td>
      <td>Tue Mar 27 23:32:05 +0000 2018</td>
    </tr>
    <tr>
      <th>487</th>
      <td>@nytimes</td>
      <td>-0.4019</td>
      <td>0.144</td>
      <td>0.856</td>
      <td>0.000</td>
      <td>H&amp;amp;M is a "fast fashion" giant. But it has ...</td>
      <td>Tue Mar 27 23:17:02 +0000 2018</td>
    </tr>
    <tr>
      <th>488</th>
      <td>@nytimes</td>
      <td>0.7003</td>
      <td>0.000</td>
      <td>0.766</td>
      <td>0.234</td>
      <td>It's pretty clear how birds, even dinosaurs, g...</td>
      <td>Tue Mar 27 23:02:03 +0000 2018</td>
    </tr>
    <tr>
      <th>489</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Evening Briefing: Here's what you need to know...</td>
      <td>Tue Mar 27 22:47:04 +0000 2018</td>
    </tr>
    <tr>
      <th>490</th>
      <td>@nytimes</td>
      <td>-0.7096</td>
      <td>0.228</td>
      <td>0.772</td>
      <td>0.000</td>
      <td>RT @jdelreal: About two dozen people are gathe...</td>
      <td>Tue Mar 27 22:32:02 +0000 2018</td>
    </tr>
    <tr>
      <th>491</th>
      <td>@nytimes</td>
      <td>-0.5994</td>
      <td>0.224</td>
      <td>0.776</td>
      <td>0.000</td>
      <td>Apple is introducing a new iPad, trying to bec...</td>
      <td>Tue Mar 27 22:17:02 +0000 2018</td>
    </tr>
    <tr>
      <th>492</th>
      <td>@nytimes</td>
      <td>0.0772</td>
      <td>0.000</td>
      <td>0.939</td>
      <td>0.061</td>
      <td>The 2020 census is a snapshot of America that ...</td>
      <td>Tue Mar 27 22:01:06 +0000 2018</td>
    </tr>
    <tr>
      <th>493</th>
      <td>@nytimes</td>
      <td>-0.6249</td>
      <td>0.221</td>
      <td>0.779</td>
      <td>0.000</td>
      <td>RT @cliffordlevy: Some Republicans fear the mi...</td>
      <td>Tue Mar 27 21:47:02 +0000 2018</td>
    </tr>
    <tr>
      <th>494</th>
      <td>@nytimes</td>
      <td>-0.8225</td>
      <td>0.364</td>
      <td>0.636</td>
      <td>0.000</td>
      <td>A fatal helicopter crash in New York’s East Ri...</td>
      <td>Tue Mar 27 21:32:51 +0000 2018</td>
    </tr>
    <tr>
      <th>495</th>
      <td>@nytimes</td>
      <td>-0.8860</td>
      <td>0.309</td>
      <td>0.691</td>
      <td>0.000</td>
      <td>He was wrongfully convicted of rape and murder...</td>
      <td>Tue Mar 27 21:16:03 +0000 2018</td>
    </tr>
    <tr>
      <th>496</th>
      <td>@nytimes</td>
      <td>-0.0772</td>
      <td>0.147</td>
      <td>0.734</td>
      <td>0.119</td>
      <td>Here are 11 movies you won’t want to miss http...</td>
      <td>Tue Mar 27 21:00:23 +0000 2018</td>
    </tr>
    <tr>
      <th>497</th>
      <td>@nytimes</td>
      <td>0.0772</td>
      <td>0.000</td>
      <td>0.933</td>
      <td>0.067</td>
      <td>If President Trump actually meets Kim Jong-un,...</td>
      <td>Tue Mar 27 20:45:08 +0000 2018</td>
    </tr>
    <tr>
      <th>498</th>
      <td>@nytimes</td>
      <td>-0.2732</td>
      <td>0.110</td>
      <td>0.890</td>
      <td>0.000</td>
      <td>“The Americans” has always been as much about ...</td>
      <td>Tue Mar 27 20:30:04 +0000 2018</td>
    </tr>
    <tr>
      <th>499</th>
      <td>@nytimes</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Roseanne believes President Trump doesn't oppo...</td>
      <td>Tue Mar 27 20:22:03 +0000 2018</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 7 columns</p>
</div>




```python
xval=[i for i in range(100)]
for i in target_user:
    plt.scatter(xval,f.groupby('Author of Tweet').get_group(i)['Compound Score'])
plt.legend(labels,bbox_to_anchor=(.3, 0, 1., 1.02),title='Media Sources')
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
plt.xticks(tick_locations, labels)
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

