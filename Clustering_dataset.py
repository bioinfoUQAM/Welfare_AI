import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import sys
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from sklearn.cluster import KMeans
from Welfare_AI_dataset import CowsDataset
import itertools
import scipy.spatial.distance as ssd
from scipy import sparse
from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

class Hierarchical_clustering:
    def __init__(self):
        pass

    @staticmethod
    def cosine_correlation(list_of_joints, index):
        #ax = plt.axes()
        matrix = scipy.spatial.distance.cdist(list_of_joints[index].dropna().T, list_of_joints[index].dropna().T, 'cosine', lambda u, v: np.sqrt(np.nansum((u - v) ** 2)))
        # sns.heatmap(np.triu(matrix), annot=True, mask=matrix)
        # plt.show()
        return matrix

    @staticmethod
    def pearson_correlation(list_of_joints, index):
        matrix = np.triu(list_of_joints[index].corr())
        # sns.heatmap(list_of_joints[index].corr(), annot=True, mask=matrix)
        # plt.show()
        return matrix

    @staticmethod
    def dissimilarity_matrix(list_of_joints, index):
        return (1 - Hierarchical_clustering.pearson_correlation(list_of_joints, index))/2


    @staticmethod
    def create_mean_matrix(list_of_matrices):  # mean matrix
        sum_matrix = 0
        for i, li in enumerate(list_of_matrices):
            sum_matrix += li
        return sum_matrix/len(list_of_matrices)

    @staticmethod
    def create_super_cosine_matrix(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        dissimilarity_matrix=[]
        for i, df in enumerate(list_of_joints):
           dissimilarity_matrix.append(Hierarchical_clustering.cosine_correlation(list_of_joints, i))
        return Hierarchical_clustering.create_mean_matrix(dissimilarity_matrix)

    @staticmethod
    def create_super_dissimilarity_matrix(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        dissimilarity_matrix = []
        for i, df in enumerate(list_of_joints):
           dissimilarity_matrix.append(Hierarchical_clustering.pearson_correlation(list_of_joints, i))
        return Hierarchical_clustering.create_mean_matrix(dissimilarity_matrix)

    @staticmethod
    def plot_comparison_clustering_trees(dissimilarity_cosine, dissimilarity_corr):
        hierarchyCosine = linkage(dissimilarity_cosine, method='average')
        hierarchyCorr = linkage(dissimilarity_corr, method='average')
        f, axes = plt.subplots(1, 2, sharey=True)
        dnCosine = dendrogram(hierarchyCosine, ax=axes[0], labels=cowNames, leaf_rotation=90, leaf_font_size=10)
        axes[0].set_ylabel('dendrogram Cosine Side2')
        dnCorr = dendrogram(hierarchyCorr, ax=axes[1], labels=cowNames, leaf_rotation=90, leaf_font_size=10)
        axes[1].set_ylabel('dendrogram Correlation Side2')

    @staticmethod
    def get_labels_clustering_cosine(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        for i, df in enumerate(list_of_joints):
            dissimilarity_cosine =Hierarchical_clustering.cosine_correlation(list_of_joints, i)
            dissimilarity_cosine = ssd.squareform(dissimilarity_cosine, checks=False)
            hierarchyCosine = linkage(dissimilarity_cosine, method='average')
            labelsCosine = fcluster(hierarchyCosine, t=2, criterion='maxclust')
            print( 'labelsCosine : ', labelsCosine)
            return labelsCosine

    @staticmethod
    def get_labels_clustering_corr(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        for i, df in enumerate(list_of_joints):
            dissimilarity_corr = Hierarchical_clustering.dissimilarity_matrix(list_of_joints, i)
            dissimilarity_corr = ssd.squareform(dissimilarity_corr, checks=False)
            hierarchyCorr = linkage(dissimilarity_corr, method='average')
            labelsCorr = fcluster(hierarchyCorr, t=2, criterion='maxclust')
            print('labelsCorr : ', labelsCorr)
            return labelsCorr
    @staticmethod
    def ground_truth_labels(original_labels, n_samples):
        return list(itertools.chain.from_iterable(itertools.repeat(x, n_samples) for x in original_labels))

    @staticmethod
    def plot_cosine_trees(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        for i, df in enumerate(list_of_joints):
            dissimilarity_cosine = Hierarchical_clustering.cosine_correlation(list_of_joints, i)
            dissimilarity_cosine = ssd.squareform(dissimilarity_cosine, checks=False)
            hierarchy_cosine = linkage(dissimilarity_cosine, method='average')
            cow_names=[CowsDataset.get_cow_name(sheet) for i, sheet in enumerate(dataset.sheet_names)]
            f, ax = plt.subplots(1, 1)
            dnCosine = dendrogram(hierarchy_cosine, ax=ax, labels=cow_names, leaf_rotation=90, leaf_font_size=10)
            joints = CowsDataset.get_joint_names(dataset)
            ax.set_title('Dendrogram Cosine' + joints[i])


    @staticmethod
    def plot_corr_trees(dataset):
        list_of_joints = CowsDataset.get_list_of_joints(dataset)
        for i, df in enumerate(list_of_joints):
            dissimilarity_matrix = Hierarchical_clustering.dissimilarity_matrix(list_of_joints, i)
            dissimilarity_matrix = ssd.squareform(dissimilarity_matrix, checks=False)
            hierarchyCorr = linkage(dissimilarity_matrix, method='average')
            cow_names = [CowsDataset.get_cow_name(sheet) for i, sheet in enumerate(dataset.sheet_names)]
            f, ax = plt.subplots(1, 1)
            dnCorr = dendrogram(hierarchyCorr, ax = ax, labels=cow_names, leaf_rotation=90, leaf_font_size=10)
            joints= CowsDataset.get_joint_names(dataset)
            ax.set_title('Dendrogram Correlation' + joints[i])
            plt.rcParams.update({'figure.max_open_warning': 0})

