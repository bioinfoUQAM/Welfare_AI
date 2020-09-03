# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:03:54 2020

@author: Yesmin
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
#import scipy.cluster.hierarchy as spc
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram


fileNames_VideoNames = pd.read_excel(r"D:\BA_Yasmine_UQAM\Dictionary_Kinematics.xlsx", "Video File -> Excel Tab Names")
fileNames_VideoNames= fileNames_VideoNames.to_numpy()
side1Names = fileNames_VideoNames[0::2]
side2Names = fileNames_VideoNames[1::2]#start from the first element ald take every 2nd one
sheets=side1Names[:,0]
cowNames=side1Names[:,1]
sheets2=side2Names[:,0]
cowNames2=side2Names[:,1]
exclude=[5,7]
Without2Cows =np.delete(cowNames, exclude)
# exclude=[1]
# Without1Cow =np.delete(cowNames, exclude)

# sheets = ['Scaled-Coord_2063_Side2', 
#           #'Scaled-Coord_5870(2)_Side2',
#           'Scaled-Coord_2078(2)_Side2', 
#           'Scaled-Coord_5327(2)_Side2', 
#           'Scaled-Coord_8527_Side2', 
#           'Scaled-Coord_8531_Side2', 
#           'Scaled-Coord_2066_Side2', 
#           'Scaled-Coord_5871_Side2', 
#           'Scaled-Coord_5865_Side2' ]
sheets = ['Scaled-Coord_2063_Side1', 
          'Scaled-Coord_5870(2)_Side1',
          'Scaled-Coord_2078(2)_Side1', 
          'Scaled-Coord_5327(2)_Side1', 
          'Scaled-Coord_8527_Side1', 
          'Scaled-Coord_8531_Side1', 
          'Scaled-Coord_2066_Side1', 
          'Scaled-Coord_5871_Side1', 
          'Scaled-Coord_5865_Side1' ]



# df=[]    
# for i, sheet in enumerate(sheets):
#     df.append(pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx",sheet))
# columns = df[1].iloc[0, 2:34].index # column names
# listColumn=[]
# for i,column in enumerate(columns):
#     for j,dCow in enumerate(df):
#         listColumn.append(dCow[column])#list containing all the columns from all the sheets/columns with the same column_name are together
# dfColumn=pd.DataFrame(listColumn).T# Dataframe containing the same column name of diffrent sheet next to each other

# list = []
# for i, column in enumerate(columns): 
#     dfJoint=dfColumn.filter(regex=column)
#     dfJoint.columns = cowNames +"   "+dfJoint.columns
#     list.append(dfJoint)

df=[]    
for i, sheet in enumerate(sheets):
    df.append(pd.read_excel(r"D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx",sheet))
columns = df[1].iloc[0, 2:34].index # column names
listColumn=[]
for i,column in enumerate(columns):
    for j,dCow in enumerate(df):
        listColumn.append(dCow[column])#list containing all the columns from all the sheets/columns with the same column_name are together
dfColumn=pd.DataFrame(listColumn).T# Dataframe containing the same column name of diffrent sheet next to each other

list = []
for i, column in enumerate(columns): 
    dfJoint=dfColumn.filter(regex=column)
    dfJoint.columns = cowNames +"   "+dfJoint.columns
    list.append(dfJoint)
correlationList=[]
correlationHeatmaps=[]
distanceMatrix=[]  
dissimilarityCosine=[]
dissimilarityCorrelation=[]
from scipy import sparse
from sklearn.metrics import pairwise_distances
#from scipy.spatial.distance import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

sumCosine=0
sumCorr=0
   
for i,li in enumerate(list):
    matrix= np.triu(li.corr())
    ax = plt.axes()
    correlationList.append(li.corr())
    distanceMatrix.append(scipy.spatial.distance.cdist(li.dropna().T,li.dropna().T,'cosine',lambda u, v: np.sqrt(np.nansum((u-v)**2))))
    dissimilarityCosine.append(1- distanceMatrix[i])
    dissimilarityCorrelation.append(1-correlationList[i])
    sumCosine= sumCosine + dissimilarityCosine[i]
    sumCorr= sumCorr + np.asarray(dissimilarityCorrelation[i])
    #distanceMatrix.append(pairwise_distances(li,'cosine'))
    # correlationHeatmaps.append( sns.heatmap(li.corr(), center=0, annot = True,mask= matrix, vmin=-1, vmax=1))# list of correlation matrices
    # ax.set_title(columns[i]+ " Side 1", fontsize= 40)
    # plt.show()
lenCosine=len(dissimilarityCosine)
lenCorr=len(dissimilarityCorrelation)
superDissimilarityCosine= sumCosine/lenCosine  
superDissimilarityCorr= sumCorr/lenCorr
hierarchyCosine=linkage(superDissimilarityCosine, method='average')
hierarchyCorr=linkage(superDissimilarityCorr, method='average')
f, axes = plt.subplots(1, 2, sharey= True)
dnCosine=dendrogram(hierarchyCosine,ax=axes[0])
axes[0].set_ylabel('dendrogram Cosine Side1')
dnCorr= dendrogram(hierarchyCorr, ax=axes[1])
axes[1].set_ylabel('dendrogram Correlation Side1 ')
labelsCosine=fcluster(hierarchyCosine,t= 2, criterion='maxclust')
labelsCorr=fcluster(hierarchyCorr,t= 2, criterion='maxclust')#labels of each cow	
#g = sns.clustermap(corr,  standard_scale =1, robust=False, method='average')
#correlationMatrix = np.reshape(correlationList, (8,4)) # 2D array of correlation matrices
# a=[]
# hierarchy=[] 
# labels=[] 
# k=2
# for i, corr in enumerate(correlationList):
#     dissimilarity= 1-corr
#     hierarchy.append(linkage(dissimilarity, method='average'))
#     fig= plt.figure(figsize=(25,10))
#     dn=dendrogram(hierarchy[i])
#     plt.show()
#     labels.append( fcluster(hierarchy[i],t= 2, criterion='maxclust'))#labels of each cow	
#     g = sns.clustermap(corr,  standard_scale =1, robust=False, method='average')
#     fig = plt.gcf()
 

