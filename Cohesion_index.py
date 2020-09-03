import sys
import logging
import csv
import numpy as np
from sklearn import metrics
import seaborn as sns
import pandas as pd
# from matplotlib import rcParams

# rcParams['font.size'] = 50
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman', 'Times']
# rcParams['xtick.labelsize'] = FONT_SIZE
# rcParams['ytick.labelsize'] = FONT_SIZE


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


class Cohesion_analysis:
    def __init__(self):
        pass

    @staticmethod
    def minInterDistance(ligneDistances, labels, i):
        mask = labels != labels[i]
        mi = np.amin(ligneDistances[mask])
        return mi

    @staticmethod
    def cohesionItem(ligneDistances, Min, labels, i):
        mask = labels == labels[i]
        mask[i] = False  # pas de comparaison avec soi-même
        ci = ligneDistances[mask] <= Min[i]
        if len(ci) == 0:
            return 0.5
        return np.sum(ci) / float(len(ci))

    @staticmethod
    def cohesion_samples(XouD, labels, metric='cosine'):
        n = labels.shape[0]
        X = metrics.pairwise_distances(XouD, metric=metric)
        Min = np.array([Cohesion_analysis.minInterDistance(XouD[i], labels, i)
                        for i in range(n)])
        coh = np.array([Cohesion_analysis.cohesionItem(XouD[i], Min, labels, i)
                        for i in range(n)])
        return coh

    @staticmethod
    def cohesion_score(X, labels, metric='cosine'):
        return np.mean(Cohesion_analysis.cohesion_samples(X, labels, metric=metric))

    @staticmethod
    def visualize_clustering_analysis(method, sample_values, avg, cluster_labels, n_clusters):
        """
        

        Parameters
        ----------
        method : TYPE String
            DESCRIPTION. Silhouette or Cohesion
        sample_values : TYPE 
            DESCRIPTION. 
        avg : TYPE
            DESCRIPTION.
        cluster_labels : TYPE 1D array
            DESCRIPTION. 
        n_clusters : TYPE int
            DESCRIPTION. cluster number

        Returns
        -------
        None.

        """
        y_lower = 10
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
        for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_values = sample_values[cluster_labels == i]

                ith_cluster_values.sort()

                size_cluster_i = ith_cluster_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)

                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

        #ax1.set_title("The cohesion plot for the various clusters.", fontsize=10)
        ax1.set_xlabel( method+ " coefficient values", fontsize= 20)
        ax1.set_ylabel("Cluster label", fontsize= 20)

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=avg, color="red", linestyle="--")

        #ax1.set_yticks([50,100,150,200,250,300,350,400,450],minor=True)  # Clear the yaxis labels / ticks
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1], minor=True)
        plt.suptitle(( method + "analysis for KMeans Corr distance clustering on 450 sample data Side1 "
                      "with NRS n_clusters = %d" % n_clusters),
                     fontsize=20, fontweight='bold')
        plt.show()

    @staticmethod        
    def plot_indices(X, silh_s, silhouette_avg, cohe_s, cohesion_avg,
                     n_clusters, cluster_labels, nomClasse, titre, mds):
        # Create a subplot with 1 row and 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(28, 14) # ???
    
        ###################################
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1 to 1
        ax1.set_xlim([np.amin(silh_s) - 0.1, 1.05])
    
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        #ax1.set_xticks(np.arange(-1.0, 1.0, 0.2))
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                silh_s[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(silhouette_avg + 0.05, y_lower + 0.5 * size_cluster_i,
                     nomClasse[i])
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax1.set_title("Silhouette")
        ax1.set_xlabel("indice silhouette")
        ax1.set_ylabel("clusters")
        
        # The vertical line for average silhoutte score of all the values
        leg = ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                          label="moyenne {:0.2f}".format(silhouette_avg))
        ax1.legend(loc="lower right", fontsize='small', framealpha=0.2)
        
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        ###################################
        # The 2nd subplot is the cohesion plot
        # The cohesion coefficient can range from 0 to 1
        ax3.set_xlim([-0.05, 1.05])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax3.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        #ax3.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #ax3.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                cohe_s[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.spectral(float(i) / n_clusters)
            ax3.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax3.text(cohesion_avg + 0.05, y_lower + 0.5 * size_cluster_i,
                     nomClasse[i])
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        ax3.set_title("Cohesion")
        ax3.set_xlabel("indice de cohesion")
        ax3.set_ylabel("clusters")
        
        # The vertical line for average silhoutte score of all the values
        leg = ax3.axvline(x=cohesion_avg, color="red", linestyle="--",
                          label="moyenne {:0.2f}".format(cohesion_avg))
        ax3.legend(loc="lower right", fontsize='small', framealpha=0.2)
        ax3.set_yticks([])  # Clear the yaxis labels / ticks
        #ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        ###################################
        # 3rd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        #ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        #            c=colors)
        
        ax2.scatter(mds[:, 0], mds[:, 1], marker='o', edgecolors='face',
                    s=40, c=colors)
        # Labeling the clusters
        #centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        #ax2.scatter(centers[:, 0], centers[:, 1],
        #            marker='o', c="white", alpha=1, s=200)
        
        #for i, c in enumerate(centers):
        #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
        
        ax2.set_title("Multidimensional Scaling")
        ax2.set_xlabel("composante 1")
        ax2.set_ylabel("composante 2")
        ax2.set_yticks([])  # Clear the yaxis ticks
        ax2.set_xticks([])  # Clear the xaxis ticks
        plt.suptitle(titre, fontsize=14, fontweight='bold')
        if matplotlib.get_backend() == 'Qt4Agg':
            # avec autre backend, il faudrait trouver une autre recette pour
            # maximiser la fenêtre
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()  
        plt.show()

# def main():
    
# df1 = pd.DataFrame(pd.read_excel(r"D:\BA_Yasmine_UQAM\Welfare_AI\dataset\PMSHBV01_learnData.csv")) 
# df2 = pd.DataFrame(pd.read_excel("D:\BA_Yasmine_UQAM\Welfare_AI\dataset\PMSHIV02_learnData.csv"))
# f = open(inFile, 'r')
# reader = csv.reader(f)

# # entete
# entete = next(reader)
# nbChamps = len(entete)
# assert nbChamps >= 3, "Nombre de champs insuffisant dans {}".format(sys.argv[1])
# type_de_classe = entete[-1]

# nomClasse = [] # nom de la classe
# ids = []       # nom de chaque exemple
# classes = []   # numéro de la classe de chaque exemple
# data = []      # les données liste de liste des attributs de chaque exemple


# for ligne in reader:
#     if len(ligne) == 0: continue
#     assert len(ligne) == nbChamps, "ligne invalide: {}".format(ligne)
#     ids.append(ligne[0])
#     if ligne[-1] not in nomClasse :
#         nomClasse.append(ligne[-1])
#     classes.append(nomClasse.index(ligne[-1]))
#     data.append(ligne[1:nbChamps-1])
    
# f.close()

# df = pd.read_csv("D:\BA_Yasmine_UQAM\Welfare_AI\dataset\PMSHBV01_learnData.csv")
#     plot_all_markers(df, sheet)

#     # xls = xlrd.open_workbook(r'D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx', on_demand=True)
#     # for sheet in xls.sheet_names():
#     #     df = pd.read_excel('D:\BA_Yasmine_UQAM\Plot\ScaledCoordinates_Post-Trial.xlsx', sheet)
#     #     plot_all_markers(df, sheet)


# if __name__ == '__main__':
#     main()
