import numpy as np
import pandas as pd
import time
import datetime as dt
import cPickle as pickle

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
weather = weather.interpolate()
weather = weather.ix[dt_index,:]

weather = weather.ix[['TempM', 'DewpointM', 'PressureM', 'WindChillM', 'HeatIndexM']


weather.to_pickle('weather.pkl')

steam = pd.read_csv('RUDINSERVER_CURRENT_STEAM_DEMAND_FX70.csv')
steam.columns = ['ID', 'TIMESTAMP', 'Steam']
steam.dropna(inplace=True)
N_steam, P_steam = steam.shape
steam.index = [x for x in xrange(N_steam)]

dt_fmt = '%Y-%m-%d %H:%M'
steam_index = [dt.datetime.strptime(ts[:16], dt_fmt) for ts in steam['TIMESTAMP']]
steam.index = steam_index
steam = steam.ix[:, 'Steam']
steam.to_pickle('steam.pkl')

data = pd.DataFrame(steam).join(weather)
data.to_pickle('data.pkl')

'''
add day of week
add season
'''
data['dayofweek'] = [d.weekday() for d in data.index]

#time lag - shifting time periods

data['lag1'] = data['Steam'].shift(-1)
data['lag2'] = data['Steam'].shift(-2)
data['lag3'] = data['Steam'].shift(-3)
data['lag4'] = data['Steam'].shift(-4)












