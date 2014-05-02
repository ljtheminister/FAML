import numpy as np
import pandas as pd
import time
import datetime as dt
import cPickle as pickle
from numpy import exp
from workalendar.america import UnitedStates

### Weather Covariates
wx_obs = pd.read_csv('[Weather].dbo.[Observations_History].csv', delimiter=',', header=None)
names = ['Run_DateTime', 'Date' ,'UTC_Date', 'TempA', 'TempM', 'DewPointA', 'DewPointM','Humidity','WindSpeedA','WindSpeedM','WindGustA','WindGustM','WindDir','VisibilityA','VisibilityM','PressureA','PressureM', 'WindChillA', 'WindChillM', 'HeatIndexA','HeatIndexM', 'PrecipA', 'PrecipM', 'Condition', 'Fog', 'Rain', 'Snow', 'Hail', 'Thunder', 'Tornado', 'unknown']

wx_obs.columns = names
wx_obs = wx_obs.dropna(axis=0, how='any', subset=['Date'])
N_wx, P_wx = wx_obs.shape

dt_fmt = '%Y-%m-%d %H:%M:%S.%f'
dt_wx = [dt.datetime.strptime(s[:-1], dt_fmt) for s in wx_obs['Date']]
wx_obs.index = dt_wx

dt_index = pd.date_range('1/1/2011 01:45:00', periods=N_wx*4, freq='15min')
dt_index = [x.to_datetime() for x in dt_index]

N_idx = len(dt_index)
empty_df = np.empty((N_idx, P_wx))
empty_df[:] = np.NAN
empty_df = pd.DataFrame(data=empty_df, index=dt_index, columns=names)

weather = pd.concat([wx_obs, empty_df])
weather.sort_index(inplace=True)

weather.replace(to_replace=[-9999], value=[None], inplace=True)
weather['index'] = weather.index
weather.drop_duplicates(cols='index', take_last=True, inplace=True)
del weather['index']

weather = weather.ix[:, ['TempM', 'DewPointM', 'Humidity', 'WindSpeedM', 'WindGustM', 'VisibilityM', 'PressureM', 'WindChillM', 'HeatIndexM', 'PrecipM']]

for col in ['WindGustM', 'WindChillM', 'HeatIndexM', 'PrecipM']:
    weather[col].fillna(0, inplace=True)

weather = weather.interpolate()
weather = weather.ix[dt_index,:] # get rows on proper datetimes after interpolation
weather['Humidex'] = weather['TempM'] + 5./9.*(6.11*exp(5417.7530*(1./273.16 - 1./(weather['DewPointM']+273.15)))-10)

# holidays!

### Steam
steam = pd.read_csv('RUDINSERVER_CURRENT_STEAM_DEMAND_FX70.csv')
steam.columns = ['ID', 'TIMESTAMP', 'Steam']
steam.dropna(inplace=True)
dt_fmt = '%Y-%m-%d %H:%M'
steam_index = [dt.datetime.strptime(ts[:16], dt_fmt) for ts in steam['TIMESTAMP']]
steam.drop(['ID', 'TIMESTAMP'], axis=1, inplace=True)
steam.index = steam_index
steam['index'] = steam.index
steam.drop_duplicates(cols='index', take_last=True, inplace=True)
del steam['index']
steam_index = pd.date_range('12/20/2011 15:00:00', periods=46568, freq='15min')
new_indices = set(steam_index) - set(steam.index)
empty_steam = pd.DataFrame(pd.Series(index=new_indices))
empty_steam.columns = ['Steam']
steam = pd.concat([steam, empty_steam])
steam.sort_index(inplace=True)
steam = steam.interpolate()
steam = steam.ix[steam_index, 'Steam']

# merge weather and steam
data = weather.join(steam, how='right')
data['dayofweek'] = [d.weekday() for d in data.index]

from sklearn.preprocessing import OneHotEncoder
n_values = np.repeat(7, len(data['dayofweek']))
N = data.shape[0]

enc = OneHotEncoder(n_values=n_values)
y = enc.fit_transform(np.matrix(data['dayofweek'])).toarray().reshape((N,7))
Y = pd.DataFrame(y)
Y.index = data.index
data = data.join(Y)

#time lag - shifting time periods
data['lag1'] = data['Steam'].shift(1)
data['lag2'] = data['Steam'].shift(2)
data['lag3'] = data['Steam'].shift(3)
data['lag4'] = data['Steam'].shift(4)
data['lag5'] = data['Steam'].shift(5)
data['lag6'] = data['Steam'].shift(6)
data['lag7'] = data['Steam'].shift(7)
data['lag8'] = data['Steam'].shift(8)

'''
data.to_pickle('data.pkl')
data.to_csv('data.csv')
data = pd.read_pickle('data.pkl')
'''

cal = UnitedStates()
fed_holidays = []
fed_holidays += cal.holidays(2011)
fed_holidays += cal.holidays(2012)
fed_holidays += cal.holidays(2013)
fed_holidays += cal.holidays(2014)

fed_holidays_indices = []
for h in fed_holidays:
    for d in pd.date_range(h[0], periods=96, freq='15min'):
        fed_holidays_indices.append(d)

data['federal_holiday'] = [0 for t in data.index]
fh_idx = list(set(fed_holidays_indices)&set(data.index))
data.ix[fh_idx, 'federal_holiday'] = np.ones(len(fh_idx))

data = data.ix[8:,:]
data.to_pickle('data.pkl')
data.to_csv('data.csv')
