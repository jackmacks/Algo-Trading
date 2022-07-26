
from logging import raiseExceptions
import numpy as np 
import pandas as pd 
import requests 
import sys
import math 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('sp_500_stocks.csv')
stocks = stocks[~stocks['Ticker'].isin(['DISCA', 'HFC','VIAC','WLTW'])]

# Function sourced from 
# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]   
        
symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []
for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

my_columns2 = ['Ticker', 'Price', 'Five-Year Price Return', 'Number of Shares to Buy']


five_year_df = pd.DataFrame(columns=my_columns2)

for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        five_year_df = five_year_df.append(
                                        pd.Series([symbol, 
                                                   data[symbol]['quote']['latestPrice'],
                                                   data[symbol]['stats']['year5ChangePercent'],
                                                   'N/A'
                                                   ], 
                                                  index = my_columns2), 
                                        ignore_index = True)  

five_year_df.sort_values('Five-Year Price Return', ascending = False, inplace = True)
five_year_df = five_year_df[:51]
five_year_df.reset_index(drop = True, inplace = True)



args = sys.argv[1:]
portfolio_size = args[0]

position_size = float(portfolio_size) / len(five_year_df.index)

for i in range(0, len(five_year_df['Ticker'])):
    five_year_df.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / five_year_df['Price'][i])


a = list(five_year_df['Ticker'])[0]
url = f'https://sandbox.iexapis.com/stable/stock/market/batch?types=chart&symbols={a}&token={IEX_CLOUD_API_TOKEN}'
_data = requests.get(url).json()

ml_data = []

for i in range(len(_data[a]['chart'])):
    ml_data.append(_data[a]['chart'][i]['close'])
train = ml_data[:19]

label = []
j = 0
for i in train:
    while j<18:
        if i > train[j+1]:
            label.append('SHORT')
            
        elif i < train[j+1]:
            label.append('LONG')
            
        j+=1   
    

z = zip(train,label)
X = np.array(list(z))

clf1 = RandomForestClassifier()
clf2 = LinearSVC()
clf3 = LogisticRegression()

clf1.fit(X[:,0].reshape(-1,1), X[:,1])
clf2.fit(X[:,0].reshape(-1,1), X[:,1])
clf3.fit(X[:,0].reshape(-1,1), X[:,1])

p1 = clf1.predict([train[18:]])
p2 = clf2.predict([train[18:]])
p3 = clf3.predict([train[18:]])

if p1 == p2: 
    print(f"{p3} {five_year_df['Number of Shares to Buy'][1]} {a}")
elif p2 == p3:
    print(f"{p3} {five_year_df['Number of Shares to Buy'][1]} {a}")
elif p3 == p1:
    print(f"{p3} {five_year_df['Number of Shares to Buy'][1]} {a}")

