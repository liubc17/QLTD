import torch
from sklearn import mixture
from typing import Tuple
import numpy as np
import sklearn
from Logger import Logger

def get_gmm_machine(training_set: torch.Tensor, k: int, n_iters: int) -> sklearn.mixture._gaussian_mixture.GaussianMixture:
    """Training_set is a row matrix, each row is a subvector."""
    gmm_machine = mixture.GaussianMixture(n_components=k,max_iter=n_iters)
    gmm_machine.fit(training_set.cpu().numpy())

    return gmm_machine


def get_prob(training_set: torch.Tensor, gmm_machine) -> np.ndarray:
    """prob is a row matrix, size = num_of_subvector * k, each row is the assignment of a subvector"""
    prob = gmm_machine.predict_proba(training_set.cpu().numpy())
    return prob


# def get_labels(training_set: torch.Tensor, gmm_machine) -> np.ndarray:
#     labels = gmm_machine.predict(training_set)
#     return labels


def get_centroids(training_set: torch.Tensor, k: int) -> torch.Tensor:
    pass


def GMM(training_set: torch.Tensor, k: int, n_iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    training_set_cpu = training_set.cpu().numpy()
    gmm_machine = get_gmm_machine(training_set, k, n_iters)
    prob = get_prob(training_set,gmm_machine)
    # make code
    code = torch.from_numpy(np.float32(prob.copy()))
    # make codebook
    num_of_trainingset = training_set.size(0)
    ## labels = get_labels(training_set, gmm_machine)
    centroids = []
    for index_centroid in range(k):
        weights = prob[:,index_centroid]
        sum = torch.zeros(1, training_set.size(1))
        for index_sample in range(num_of_trainingset):
            sum += training_set_cpu[index_sample] * weights[index_sample]
        centroid = sum / np.sum(weights)
        # print('centroid_size',centroid.size())
        centroids.append(centroid)
    codebook = torch.cat(centroids,dim=0)
    # print('codebook',codebook)
    logger = Logger()
    logger.logger.info('codebook_size:{}'.format(codebook.size()))
    logger.logger.info('code_size:{}'.format(code.size()))
    print('codebook_size:',codebook.size())
    print('code_size:',code.size())
    return codebook,code
