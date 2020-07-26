from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from common.spectogram.spectrogram_png_safer import save_create_dendrogram
from common.utils.logger import *
from common.clustering.k_medoids import KMedoids

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

import numpy as np
from common.clustering import dominantset as ds


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
        elif algorithm == 'DS':
            predicted_clusters = []

            a = np.asarray(set_of_speakers[0])
            #a = embeddings
            dos = ds.DominantSetClustering(feature_vectors=embeddings, speaker_ids=a,
                                           metric='cosine', dominant_search=False,
                                           epsilon=epsilon, cutoff=cutoff)

            dos.apply_clustering()
            labels = dos.evaluate()


            #print("MR\t\tARI\t\tACP")
            #print("{0:.4f}\t\t{1:.4f}\t\t{2:.4f}".format(mr, ari, acp))  # MR - ARI - ACP
            predicted_clusters.append(labels)
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