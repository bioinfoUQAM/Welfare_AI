# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:32:55 2020

@author: Yesmin
"""
import pandas as pd
import matplotlib.pyplot as plt
from fitter import Fitter
import numpy as np
import scipy
from scipy import optimize
from scipy.signal import argrelextrema

fileNames_VideoNames = pd.read_excel(r"D:\BA_Yasmine_UQAM\Dictionary_Kinematics.xlsx", "Video File -> Excel Tab Names")
fileNames_VideoNames= fileNames_VideoNames.to_numpy()
side1Names = fileNames_VideoNames[0::2]
side2Names = fileNames_VideoNames[1::2]#start from the first element ald take every 2nd one
sheets=side1Names[:,0]
# for sheet in sheets:
#     df = pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx", sheet)
#     #plot_all_markers_subplots(df, sheet)
#     plot_all_markers(df, sheet)

sheets2=side2Names[:,0]
df = [pd.DataFrame(pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx",sheet).iloc[:, 0:34]) for x,sheet in enumerate(sheets)]  
data= df[0]['Z-E']
columns = df[1].iloc[0, 2:34].index 
param = scipy.stats.norm.fit(data.dropna())
random_samples = scipy.stats.norm.rvs(param[0], param[1], size=data.dropna().shape[0])
sorted=np.sort(random_samples)
plt.plot(sorted)
plt.plot(data)
plt.show()
ax=data.plot.hist(bins=10)
# f = Fitter(df[0]['Z-E'])       
# f.fit()
# print(f)
# x = np.linspace (0, 300, 10000)
# y = df[0]['X-E']
# peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
# dips = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1
# plt.plot (x, y)
# plt.plot (x[peaks], y[peaks], 'o')
# plt.plot (x[dips], y[dips], 'o')
# plt.show()
local_maxima = argrelextrema(data.to_numpy(), np.greater_equal, order = 5, mode = 'clip')[0]
local_maxima = [local_maxima]
local_minima = argrelextrema(data.to_numpy(), np.less_equal, order = 5, mode = 'clip')[0]
local_minima=[local_minima]
local_maxima.extend(local_minima)
import itertools
extrema = np.sort(list(itertools.chain.from_iterable(local_maxima)))
random_samples = []
sorted = []
sequences= extrema
for i, sequence in enumerate(sequences):
    if (i+1) < len(sequences):
        param = scipy.stats.norm.fit(data[sequences[i]+1: sequences[i+1]+1])
        sorted.extend(np.sort(scipy.stats.norm.rvs(param[0], param[1], size=data.dropna().shape[0])))
        # sorted.extend(np.sort(random_samples[i]))
    else:
        pass

plt.plot(sorted)
plt.plot(data)
plt.show()
ax = data.plot.hist(bins=10)

print('localmax order 3', local_maxima)
print('localmin order 3', local_minima)
# local_maxima = np.array(argrelextrema(data.to_numpy(), np.greater_equal, order = 10, mode = 'clip'))
# local_minima = np.array(argrelextrema(data.to_numpy(), np.less_equal, order = 10, mode = 'clip'))
# print('localmax order 10', local_maxima)
# print('localmin order 10', local_minima)
# x = df[0]['Y-E']
# prev = x[0] 
# curr = x[1] #keep track of two items together during iteration, previous and current
# # result = "increasing": ,
# # "equal" ,
# # "decreasing": ,
# result = {"increasing": [],
#           "equal": [],
#           "decreasing": [],
#           }
 


# def two_item_relation(prev, curr): #compare two items in list, results in what is effectively a 3 way flag
#  if prev <= curr:
#      return "increasing"
#  elif prev == curr:
#      return "equal"
#  else:
#      return "decreasing"


# prev_state = two_item_relation(prev, curr) #keep track of previous state
# result[prev_state].append([prev]) #handle first item of list

# x_shifted = iter(x)
# next(x_shifted) #x_shifted is now similar to x[1:]

# for curr in x_shifted: 
#  curr_state = two_item_relation(prev, curr)
#  if prev_state == curr_state: #compare if current and previous states were same.
#      result[curr_state][-1].append(curr) 
#  else: #states were different. aka a change in trend
#      result[curr_state].append([])
#      result[curr_state][-1].extend([prev, curr])
#  prev = curr
#  prev_state = curr_state

# mu = data.mean()#.values
# cov = data.cov()#.values
# number_of_sample = data.shape[0]
# generate using affine transformation
#c2 = cholesky(cov).T
# c2 = sqrtm(cov).T
# s = np.matmul(c2, np.random.randn(c2.shape[0], number_of_sample)) + mu.reshape(-1, 1)