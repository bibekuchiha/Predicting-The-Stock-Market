#!/usr/bin/env python
# coding: utf-8

# # Predicting The Stock Market
# 
# In this project, we'll be working with data from the [S&P500 Index](https://en.wikipedia.org/wiki/S%26P_500_Index).
# 
# We will be using historical data on the price of the S&P500 Index to make predictions about future prices. Predicting whether an index will go up or down will help us forecast how the stock market as a whole will perform. Since stocks tend to correlate with how well the economy as a whole is performing, it can also help us make economic forecasts.
# 
# We'll be working with a csv file containing index prices. Each row in the file contains a daily record of the price of the S&P500 Index from 1950 to 2015. The dataset is stored in sphist.csv.
# 
# The columns of the dataset are:
# 
# - Date -- The date of the record.
# - Open -- The opening price of the day (when trading starts).
# - High -- The highest trade price during the day.
# - Low -- The lowest trade price during the day.
# - Close -- The closing price for the day (when trading is finished).
# - Volume -- The number of shares traded.
# - Adj Close -- The daily closing price, adjusted retroactively to include any corporate actions. Read more here.
# 
# We'll be using this dataset to develop a predictive model. We'll train the model with data from 1950-2012, and try to make predictions from 2013-2015.

# In[16]:


import pandas as pd

df = pd.read_csv('sphist.csv')
df.head()


# Now, lets convet the Date Column in to datetime format

# In[17]:


from datetime import datetime

df['Date'] = pd.to_datetime(df['Date'])


#  This will generate dataframe that tells you if each item in the Date column is after 2015-04-01

# In[18]:


df[df['Date'] > datetime(year = 2015, month =4, day=1)]


# Sorting the dataframe on the Date column. It's currently in descending order, but we'll want it to be in ascending order for some of the next steps. 

# In[19]:


df = df.sort_values('Date').reset_index()
df.head()


# In[20]:


df.info()


# ##  Generating indicators
# 
# Datasets taken from the stock market need to be handled differently than datasets from other sectors when it comes time to make predictions. In a normal machine learning exercise, we treat each row as independent. Stock market data is sequential, and each observation comes a day after the previous observation. Thus, the observations are not all independent, and you can't treat them as such.
# 
# This means we have to be extra careful to not inject "future" knowledge into past rows when we do training and prediction. Injecting future knowledge will make our model look good when we are training and testing it, but will make it fail in the real world. This is how many algorithmic traders lose money.

# In[21]:


#Calculate the mean for the past 5, 30, 365 days
df['day_5'] = df['Close'].rolling(5).mean().shift(1)
df['day_30'] = df['Close'].rolling(30).mean().shift(1)
df['day_365'] = df['Close'].rolling(365).mean().shift(1)

#Calculate the STD for the past 5, 365 days
df['std_5'] = df['Close'].rolling(5).std().shift(1)
df['std_365'] = df['Close'].rolling(365).std().shift(1)

#Calculate the mean volume for the past 5, 365 days
df['day_5_volume'] = df['Volume'].rolling(5).mean().shift(1)
df['day_365_volume'] = df['Volume'].rolling(365).mean().shift(1)

#Calculate the STD of the average volume over the past five days
df['5_volume_std'] = df['day_5_volume'].rolling(5).std().shift(1)


# In[22]:


df.head(10)


# In[23]:


df.tail(10)


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(df['Date'],df['Open'])
plt.xlabel('Date')
plt.ylabel('The opening price of the trading days')
plt.show()


# In[56]:


plt.plot(df['Date'],df['Close'])
plt.xlabel('Date')
plt.ylabel('The closing price of the trading days')
plt.show()


# In[57]:


plt.plot(df['Date'],df['Volume'])
plt.xlabel('Date')
plt.ylabel(' The number of shares traded')
plt.show()


# In[58]:


plt.plot(df['Date'],df['day_365'])
plt.xlabel('Date')
plt.ylabel('The average price for the past 365 trading days')
plt.show()


# In[59]:


plt.plot(df['Date'],df['std_365'])
plt.xlabel('Date')
plt.ylabel('The variance price for the past 365 trading days')
plt.show()


# ##  Splitting up the data
# 
# Since we are computing indicators that use historical data, there are some rows where there isn't enough historical data to generate them. Some of the indicators use 365 days of historical data, and the dataset starts on 1950-01-03. Thus, any rows that fall before 1951-01-03 don't have enough historical data to compute all the indicators. We will need to remove these rows before you split the data.

# In[24]:


df = df[df['Date'] > datetime(year = 1951, month = 1, day = 3)]
df.head()


# Now we have to remove any rows with NaN values 

# In[25]:


df.isnull().sum()


# In[26]:


df = df.dropna(axis = 0)


# In[27]:


df.isnull().sum()


# Let's now generate two new dataframes to use in making our algorithm. train should contain any rows in the data with a date less than 2013-01-01. test should contain any rows with a date greater than or equal to 2013-01-01.

# In[28]:


train = df[df['Date'] < datetime(year = 2013, month = 1, day = 1)]
test = df[df['Date'] >= datetime(year = 2013, month = 1, day = 1)]


# In[31]:


train.shape


# In[32]:


test.shape


# In[33]:


train.columns


# ## Making predictions
# Now, you can define an error metric, train a model using the train data, and make predictions on the test data.
# 
# We are now ready to train the algorithum, make predictions and calculate the Mean Squared Error. Our target column is Close.
# 
# 

# Leaving out all of the original columns (Close, High, Low, Open, Volume, Adj Close, Date) when training the model. These all contain knowledge of the future that you don't want to feed the model. Use the Close column as the target.

# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

features = ['day_5', 'day_30', 'day_365', 'std_5', 'std_365', 'day_5_volume',
       'day_365_volume', '5_volume_std']
target = train['Close']

lr = LinearRegression()
lr.fit(train[features],target)
predictions = lr.predict(test[features])
mse = mean_squared_error(test['Close'], predictions)
mse


# Let's now make a prediction just one day ahead.
# 
# 

# In[38]:


train_1 = df.iloc[:-1]
test_1 = df.iloc[-1:]

lr.fit(train_1[features],train_1['Close'])
predictions_1 = lr.predict(test_1[features])
mse_1 = mean_squared_error(test_1['Close'], predictions_1)
mse_1


# In[ ]:




