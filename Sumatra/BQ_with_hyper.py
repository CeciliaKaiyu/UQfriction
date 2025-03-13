import torch
from torch import nn

def BQ(X, Y, Hyper, kernel, KM, IE, nugget):
    """
    Bayesian Quadrature (BQ) estimator for level l.

    :param X: Tensor of shape (N, D) - input data points
    :param Y: Tensor of shape (N, ) - observed function values
    :param Hyper: Hyperparameters for the kernel
    :param kernel: Function that computes the covariance matrix
    :param KM: Function to compute kernel mean
    :param IE: Function to compute initial error
    :param nugget: Small positive float for numerical stability
    :return: (E, V) - BQ estimate and variance
    """

    # Compute kernel matrix with nugget for numerical stability
    K = kernel(X=X, hyper=Hyper) + nugget * torch.eye(X.size(0))

    # Compute K inverse
    Inv_K = torch.inverse(K)

    # Compute kernel mean and BQ estimate
    km = KM(X=X, hyper=Hyper)
    E = km @ Inv_K @ Y

    # Compute initial error and variance
    ie = IE(hyper=Hyper)
    V = ie - km @ Inv_K @ km

    return E, V
