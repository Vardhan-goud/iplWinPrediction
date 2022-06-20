#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd


# In[46]:


match = pd.read_csv("matches.csv")
delivery = pd.read_csv("deliveries.csv")
delivery.tail()


# In[47]:


match.head()


# In[48]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df


# In[49]:


total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df


# In[50]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[51]:


match_df.head()


# In[52]:


match_df['team1'].unique()


# In[53]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[54]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[55]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[56]:


match_df.shape


# In[57]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[58]:


match_df['dl_applied'].value_counts()


# In[59]:


match_df.head()


# In[60]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[61]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[62]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[63]:


delivery_df.shape


# In[64]:


delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[65]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[66]:


delivery_df


# In[67]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[68]:


delivery_df.head()


# In[70]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")


# In[71]:


delivery_df


# In[72]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")


# In[73]:


delivery_df


# In[75]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets
delivery_df.tail()


# In[76]:


# crr = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[77]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[78]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[79]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[80]:


delivery_df


# In[81]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[83]:


final_df.shape


# In[84]:


final_df = final_df.sample(final_df.shape[0])


# In[86]:


final_df.sample()


# In[88]:


final_df.isna().sum()


# In[89]:


final_df.dropna(inplace=True)


# In[90]:


final_df = final_df[final_df['balls_left'] != 0]


# In[91]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[92]:


X_train


# In[95]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[96]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[97]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[98]:


pipe.fit(X_train,y_train)


# In[99]:


y_pred = pipe.predict(X_test)


# In[100]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[101]:


pipe.predict_proba(X_test)[10]


# In[ ]:




