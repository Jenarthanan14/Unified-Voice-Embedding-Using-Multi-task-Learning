from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from common.spectogram.spectrogram_png_safer import save_create_dendrogram
from common.utils.logger import *
from common.clustering.k_medoids import KMedoids

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

import numpy as np
#from common.clustering import dominantset as ds


def cluster_embeddings(set_of_embeddings, metric='cosine', method='complete', algorithm='AHC',cluster_count=40,
                       set_of_speakers=None, epsilon=1e-6, cutoff=-0.1):
    """
    Calculates the distance and the linkage matrix for these embeddings.

    :param set_of_embeddings: The embeddings we want to calculate on
    :param metric: The metric used for the distance and linkage
    :param method: The linkage method used.
    :return: The embedding Distance and the embedding linkage
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Cluster embeddings')

    set_predicted_clusters = []
    predicted_clusters = None
    #print("------------------>>>>>>>>>>> det embeddings list\n")
    #print(len(set_of_embeddings))

    for embeddings in set_of_embeddings:
        #print('---------------------', len(embeddings))
        #print(embeddings)
        if algorithm == 'AHC':
            embeddings_distance = cdist(embeddings, embeddings, metric)

            #print("------------------>>>>>>>>>>> distance matrix\n")
            #print(embeddings_distance.shape)

            embeddings_linkage = linkage(embeddings_distance, method, metric)

            save_create_dendrogram(embeddings_linkage)

            #print("------------------>>>>>>>>>>> threshold list creation\n")
            #print(embeddings_linkage.shape)

            thresholds = embeddings_linkage[:, 2]

            #print("------------------>>>>>>>>>>> threshold list\n")
            #print(thresholds)

            predicted_clusters = []

            for threshold in thresholds:
                predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
                predicted_clusters.append(predicted_cluster)

                # if (max(predicted_cluster) == 5):
                #     print("------------------>>>>>>>>>>> predicted cluster\n")
                #     print(predicted_cluster)

            set_predicted_clusters.append(predicted_clusters)
        elif algorithm == 'K-MEANS':
            predicted_clusters = []
            kmeans_model = KMeans(n_clusters=cluster_count).fit(embeddings)
            # Predicting the clusters
            labels = kmeans_model.predict(embeddings)
            predicted_clusters.append(labels)
        # elif algorithm == 'DS':
        #     predicted_clusters = []

        #     a = np.asarray(set_of_speakers[0])
        #     #a = embeddings
        #     dos = ds.DominantSetClustering(feature_vectors=embeddings, speaker_ids=a,
        #                                    metric='cosine', dominant_search=False,
        #                                    epsilon=epsilon, cutoff=cutoff)

        #     dos.apply_clustering()
        #     labels = dos.evaluate()


        #     #print("MR\t\tARI\t\tACP")
        #     #print("{0:.4f}\t\t{1:.4f}\t\t{2:.4f}".format(mr, ari, acp))  # MR - ARI - ACP
        #     predicted_clusters.append(labels)
        elif algorithm =='SP':
            predicted_clusters = []
            labels = SpectralClustering(n_clusters=cluster_count, assign_labels="discretize", random_state=0)\
                .fit_predict(embeddings)
            print('SP : ', labels)
            print("MAX :", max(labels))
            predicted_clusters.append(labels)

        elif algorithm == 'K-MEDOIDS':
            predicted_clusters = []
            model = KMedoids(n_clusters=cluster_count, dist_func=squared_distance)
            labels = model.fit(embeddings, plotit=False, verbose=True)
            print('K-MEDOID : ', labels)
            print("MAX :", max(labels))
            predicted_clusters.append(labels)

        set_predicted_clusters.append(predicted_clusters)
    return set_predicted_clusters


def squared_distance(data1, data2):
    '''example distance function'''
    return np.sqrt(np.sum((data1 - data2) ** 2))


def get_clusters(self, cluster_count=None, algorithm='AHC', stored=False, total_embeddings=40,
                 epsilon=1e-6, cutoff=-0.1):
    """
    Generates the predicted_clusters with the results of get_embeddings.
    All return values are sets of possible multiples.
    :return:
    checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
    """

    print("Incoming couster count inside get clusters ===========> ", cluster_count)
    #checkpoint_names, set_of_embeddings, set_of_true_clusters, embeddings_numbers = self.get_embeddings(cluster_count, stored,
                                                                                                        total_embeddings=total_embeddings)

    print("Before clustering ===========> ", cluster_count)
    set_of_predicted_clusters = cluster_embeddings(set_of_embeddings, cluster_count=cluster_count,
                                                   algorithm=algorithm, set_of_speakers=set_of_true_clusters,
                                                   epsilon=epsilon, cutoff=cutoff)

    #print("------------------>>>>>>>>>>> predicted results\n")
    #print(set_of_predicted_clusters)

    #print("------------------>>>>>>>>>>> true clusters\n")
    #print(set_of_true_clusters)

    return set_of_predicted_clusters, set_of_true_clusters

def calculate_analysis_values(predicted_clusters, true_cluster, cluster_count=None, mr_list=None, mr_dict_index=None):
    """
    Calculates the analysis values out of the predicted_clusters.

    :param predicted_clusters: The predicted Clusters of the Network.
    :param true_clusters: The validation clusters
    :return: misclassification rate, homogeneity Score, completeness score and the thresholds.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')

    #
    # print("------------------>>>>>>>>>>>  before incremental true clusters\n")
    # print(true_cluster)
    # for i in range(len(true_cluster)):
    #     true_cluster[i] += 1
    #
    # print("------------------>>>>>>>>>>>  after incremental true clusters\n")
    # print(true_cluster)

    # Initialize output
    mrs = np.ones(len(true_cluster))
    homogeneity_scores = np.ones(len(true_cluster))
    completeness_scores = np.ones(len(true_cluster))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        # Calculate different analysis's
        mrs[i] = misclassification_rate(true_cluster, predicted_cluster)
        homogeneity_scores[i] = homogeneity_score(true_cluster, predicted_cluster)
        completeness_scores[i] = completeness_score(true_cluster, predicted_cluster)
        #print("---------------------------------->>>>>>>>>>>>>>>>>>>>...")
        #print(i, predicted_cluster)
        if cluster_count is not None and ((max(predicted_cluster) == cluster_count and mr_dict_index == 'AHC') or
                                          (max(predicted_cluster) == cluster_count -1 and mr_dict_index == 'K-MEANS') or
                                          (max(predicted_cluster) == cluster_count -1 and mr_dict_index == 'SP') or
                                          (max(predicted_cluster) == cluster_count -1 and mr_dict_index == 'K-MEDOIDS')):
            if mr_dict_index is not None:
                mr_list[mr_dict_index].append(mrs[i])
            else:
                mr_list[cluster_count] = mrs[i]
        elif cluster_count is not None and mr_dict_index == "DS":
            if mr_dict_index is not None:
                mr_list[mr_dict_index].append(mrs[i])
            else:
                mr_list[cluster_count] = mrs[i]

        #print(">>>>>>>>>>>>>>>>>>>>>>>>>> Predicted and true clusters")
        #print(predicted_clusters, true_cluster)

    return mrs, homogeneity_scores, completeness_scores

def main():

    predicted_clusters, set_of_true_clusters = get_clusters(self, cluster_count=None, algorithm='AHC', stored=False, total_embeddings=40,
                 epsilon=1e-6, cutoff=-0.1)

    mrs, homogeneity_scores, completeness_scores = calculate_analysis_values(predicted_clusters,
                                                                                 set_of_true_clusters[0],
                                                                                 cluster_count, mr_list = mr_list,
                                                                                 mr_dict_index=mr_dict_index)
    return mrs, homogeneity_scores, completeness_scores

if __name__ == '__main__':
    main()


