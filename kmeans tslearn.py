# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:51:44 2020

@author: Yesmin
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt


def dfToTensor(sheets):
    data = [pd.DataFrame(pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx",sheet).iloc[:, 2:34]) for x,sheet in enumerate(sheets)]
    list_of_arrays = [np.array(df.dropna()) for df in data]
    list=[x.shape[0] for x in list_of_arrays ]
    m = max(list)
    superDataArray=[]
    for i, x in enumerate(list_of_arrays):
        superDataArray = np.zeros((m,32))
        superDataArray[: x.shape[0],:] = x
        list_of_arrays[i] = superDataArray
    tensor= np.array(list_of_arrays)
    return tensor
    
def tskmeans(tensor,metric,side, n_clusters):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,max_iter=10,random_state=0).fit_predict(tensor)
    print(metric + side ,km)
    
def main():
    
    fileNames_VideoNames = pd.read_excel(r"D:\BA_Yasmine_UQAM\Dictionary_Kinematics.xlsx", "Video File -> Excel Tab Names")
    fileNames_VideoNames= fileNames_VideoNames.to_numpy()
    side1Names = fileNames_VideoNames[0::2]
    side2Names = fileNames_VideoNames[1::2]#start from the first element ald take every 2nd one
    sheets=side1Names[:,0]
    cowNames=side1Names[:,1]
    sheets2=side2Names[:,0]
    cowNames2=side2Names[:,1]
    exclude=[5,7]
    Without2Cows =np.delete(sheets, exclude).tolist()
    exclude=[1]
    Without1Cow =np.delete(sheets2, exclude).tolist()
    
    Tensor=dfToTensor(sheets)
    tskmeans(Tensor, 'euclidean', 'Side1 all',2)
    
    Tensor=dfToTensor(Without2Cows)
    tskmeans(Tensor, 'euclidean', 'Side1',2)
    
    Tensor=dfToTensor(sheets2)
    tskmeans(Tensor, 'euclidean', 'Side2 all',2)
    
    Tensor=dfToTensor(Without1Cow)
    tskmeans(Tensor, 'euclidean', 'Side2',2)
    
    Tensor=dfToTensor(sheets)
    tskmeans(Tensor, 'dtw', 'Side1 all',2)
    
    Tensor=dfToTensor(Without2Cows)
    tskmeans(Tensor, 'dtw', 'Side1',2)
    
    Tensor=dfToTensor(sheets2)
    tskmeans(Tensor, 'dtw', 'Side2 all',2)
    
    Tensor=dfToTensor(Without1Cow)
    tskmeans(Tensor, 'dtw', 'Side2',2)





if __name__ == '__main__':
    main()  
    
    
    
    
    
    
    