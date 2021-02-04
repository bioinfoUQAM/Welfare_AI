from Welfare_AI_dataset import CowsDataset
import pandas as pd
import os
from Data_generation import GeneratedData
from Clustering_dataset import Hierarchical_clustering
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
import scipy.spatial.distance as ssd
from sklearn.cluster import KMeans
from Cohesion_index import Cohesion_analysis
import time
import seaborn as sns
from sklearn.manifold import TSNE


DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), 'Dictionary_Kinematics.xlsx')
dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
dictionary = dictionary.to_numpy()
names = CowsDataset.get_side_sheets('side1')

dataset = CowsDataset(names)

columns = CowsDataset.get_joint_names(dataset)

generated_dataset=[]
mix_dataset=[]
new_dataset = []
new_sheets = []

for i, sheet in enumerate(tqdm(dataset)):
    new_dataset.append(GeneratedData.create_generated_cow_data(dataset, i, 5)[0])
    new_sheets.append(GeneratedData.create_generated_cow_data(dataset, i, 5)[1])

joint_list = []
for j, column in enumerate(columns):
    for i, cow in enumerate(new_dataset):
        for k, instance in enumerate(new_dataset[i]):
            joint_list.append(new_dataset[i][k][column])
joint_list = pd.DataFrame(joint_list).T

instance_names = []
for i, cow in enumerate(new_dataset):
    for k, instance in enumerate(new_dataset[i]):
        instance_names.append(new_sheets[i][k])

print('length instance_names ', len(instance_names))
# list of lists each list has 9*50 cow sapme for a specific joint
joint_lists = []
for i, column in enumerate(columns):
    dfJoint = joint_list.filter(regex=column)
    dfJoint.columns = instance_names + dfJoint.columns
    joint_lists.append(dfJoint)

            
sumCosine = 0
sumCorr = 0
dissimilarityCosine = []
dissimilarityCorrelation = []
correlationList = []
for i,li in enumerate(tqdm(joint_lists)):
    matrix= np.triu(li.corr())
    correlationList.append(li.corr())
    dissimilarityCosine.append(ssd.cdist(li.dropna().T,li.dropna().T,'cosine',lambda u, v: np.sqrt(np.nansum((u-v)**2))))
    dissimilarityCorrelation.append((1-correlationList[i])/2)
    sumCosine = sumCosine + dissimilarityCosine[i]
    sumCorr = sumCorr + np.asarray(dissimilarityCorrelation[i])

lenCosine = len(dissimilarityCosine)
lenCorr = len(dissimilarityCorrelation)
print('lenCorr', lenCorr)
# a squareformed projection matrix of  the 32 distance matrixes
square_mean_cosine = sumCosine/lenCosine
square_mean_corr = sumCorr/lenCorr

f, axes = plt.subplots(1, 2, sharey=True)
f.suptitle('clustering of the generated data with 7% deviation')

# Clustering of the generated data using Cosine distance
superDissimilarityCosine = ssd.squareform(square_mean_cosine, checks=False)# convert the redundant n*n square matrix form into a condensed nC2 array
hierarchyCosine = linkage(superDissimilarityCosine, method='average')
dnCosine = dendrogram(hierarchyCosine,ax=axes[0],labels=instance_names, leaf_rotation =90, leaf_font_size= 1)
axes[0].set_ylabel('dendrogram Cosine Side2 with all the cows')
labelsCosine = fcluster(hierarchyCosine, t=2, criterion='maxclust')
#print('labelsCosine.shape', labelsCosine.shape)
print('labels Cosine ', np.split(labelsCosine,9))

# Clustering of the generated data using dissimilarity matrix from pearson correlation
superDissimilarityCorr = ssd.squareform(square_mean_corr, checks=False)# convert the redundant n*n square matrix form into a condensed nC2 array
hierarchyCorr = linkage(superDissimilarityCorr, method='average')
dnCorr = dendrogram(hierarchyCorr, ax=axes[1], labels=instance_names, leaf_rotation =90, leaf_font_size= 1)
axes[1].set_ylabel('dendrogram Correlation Side2 with all the cows')
labelsCorr = fcluster(hierarchyCorr, t=2, criterion='maxclust')
print('labels corr ', np.split(labelsCorr, 9))
plt.show()



#Silhouette and Cohesion indexes

# # real_cows_labels_cosine = Hierarchical_clustering.get_labels_clustering_cosine(dataset)
# # ground_truth_data_labels_cosine = Hierarchical_clustering.ground_truth_labels(real_cows_labels_cosine, 50)
# # silhouette_score_cosine = silhouette_score(sumCosine/lenCosine, ground_truth_data_labels_cosine, metric='precomputed')
# # silhouette_samples_cosine = silhouette_samples(sumCosine/lenCosine, ground_truth_data_labels_cosine, metric='precomputed')
# #
# # real_cows_labels_corr = Hierarchical_clustering.get_labels_clustering_corr(dataset)
# # ground_truth_data_labels_corr = Hierarchical_clustering.ground_truth_labels(real_cows_labels_corr, 50)
# # silhouette_score_corr = silhouette_score(sumCorr/lenCorr, ground_truth_data_labels_corr, metric='precomputed')
# # silhouette_samples_corr = silhouette_samples(sumCorr/lenCorr, ground_truth_data_labels_corr, metric='precomputed')

    # real_super_cosine = Hierarchical_clustering.create_super_cosine_matrix(dataset) # distance matrices of the original data
    # kCos = KMeans(n_clusters=2, random_state=0).fit_predict(real_super_cosine) #labels od the original data
    # ground_truth_data_labels_kcosine = Hierarchical_clustering.ground_truth_labels(kCos, 50)#repeat every label 50 times so that we have the truth labels of every generated data
    # #ground_truth_data_labels_kcosine = np.array(ground_truth_data_labels_kcosine)
    # #labels of the cows that from the gait scoring
    # cluster_labels = Hierarchical_clustering.ground_truth_labels([0, 1, 0, 0, 0, 1, 1, 1, 1], 50)#labels from the NRS gait scoring
    
    # silhouette_score_kcosine = silhouette_score(square_mean_cosine, cluster_labels, metric='precomputed')
    # silhouette_samples_kcosine = silhouette_samples(square_mean_cosine, cluster_labels, metric='precomputed')


    # n_clusters = 2
# real_dissimilarity_matrix = Hierarchical_clustering.create_super_dissimilarity_matrix(dataset)
    # kCorr =KMeans(n_clusters=2, random_state=0).fit_predict(real_dissimilarity_matrix)
    # ground_truth_data_labels_kcorr = Hierarchical_clustering.ground_truth_labels(kCorr, 50)
    # silhouette_score_kcorr = silhouette_score(sumCorr/lenCorr, ground_truth_data_labels_kcorr, metric='precomputed')
    # silhouette_samples_kcorr = silhouette_samples(sumCorr/lenCorr, ground_truth_data_labels_kcorr, metric='precomputed')
    # sample_silhouette_values = np.array(silhouette_samples_kcorr)
    # sample_silhouette_avg = silhouette_score_kcorr
    # cluster_labels = np.array(ground_truth_data_labels_kcorr)
    # cluster_labelsNRS = np.array(Hierarchical_clustering.ground_truth_labels([0, 1, 0, 0, 0, 1, 1, 1, 1], 50))#labels from the NRS gait scoring
    # Cohesion_analysis.visualize_clustering_analysis('Silhouette ', sample_silhouette_values, sample_silhouette_avg, cluster_labelsNRS, n_clusters)    

# sample_silhouette_values = np.array(silhouette_samples_kcosine)
# sample_silhouette_avg = silhouette_score_kcosine
# cluster_labels = np.array(ground_truth_data_labels_kcosine)


# Cohesion_analysis.visualize_clustering_analysis('Silhouette ', sample_silhouette_values, sample_silhouette_avg, cluster_labels, n_clusters)

# sample_cohesion_values = Cohesion_analysis.cohesion_samples(square_mean_cosine, cluster_labels, metric='precomputed')
# cohesion_avg = Cohesion_analysis.cohesion_score(square_mean_cosine, cluster_labels, metric='precomputed')
# Cohesion_analysis.visualize_clustering_analysis('Cohesion ', sample_cohesion_values, cohesion_avg, cluster_labels, n_clusters)
# kmeanModel = KMeans(n_clusters = 2)
# kmeanModel.fit(real_dissimilarity_matrix)


def tSNE_visualiation(square_mean_cosine, instance_names):
    #square_mean_cosine = real_dissimilarity_matrix
    cow_names = instance_names
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=1500)
    tsne_results = tsne.fit_transform(square_mean_cosine)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    fig,ax = plt.subplots(1)
    # for i,cow_name in enumerate(cow_names):
    # for i in range(2):
    x = tsne_results[0,:]
    y = tsne_results[1,:]
    ax.scatter(x, y)
    ax.set_title('t-SNE Visualization of the generated data side 1')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
        # #ax.annotate(
        #     cow_name,
        #     fontsize='small',
        #     xy=(x, y),
        #     xytext=(5, 2),
        #     textcoords='offset points',
        #     ha='center',
        #     va='bottom')
    plt.show()


tSNE_visualiation(square_mean_cosine, instance_names)        
    
def plot_clusters(n_clusters, cluster_labels, tabLims, mds, cList, titre, xLabel, yLabel, outFig):

   fig, ax = plt.subplots()
   fig.set_size_inches(31, 31)
   
   colors = [cList[x] for x in cluster_labels]
 
   if len(tabLims) > 0 :
     ax.set_xlim([tabLims[0][0], tabLims[0][1]])
     ax.set_ylim([tabLims[1][0], tabLims[1][1]])
   
   ax.scatter(mds[:, 0], mds[:, 1], marker='o', edgecolors='face',
           s=2500, c=colors, alpha=0.4, linewidths=0)
   
   
   ax.set_title(titre)
   ax.set_xlabel(xLabel, fontsize=FONT_SIZE)
   ax.set_ylabel(yLabel, fontsize=FONT_SIZE)
   


   plt.show()
   fig.savefig(outFig)   
    
    
    
    
    
    
    
