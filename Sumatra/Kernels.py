# This script defines kernel functions that we can use in Tsunami example
# sklearn-rbf kernel takes the form k(x,y)=exp(-(x-y)**2/(2*length-scale**2))
# thus: the hyperparameter "hyper" in Gauss kernel we define equals 2 * sklearn-rbf-lengthscale**2

import torch

def Gauss(X, hyper):
    """
    Computes the Gaussian covariance matrix using a squared exponential kernel.

    :param X: Tensor of shape (N, ) - input data
    :param hyper: float - squared length-scale (length_scale**2)
    :return: Tensor of shape (N, N) - Gaussian covariance matrix
    """

    # Compute pairwise squared Euclidean distance
    N = X.size()[0]
    X = X.unsqueeze(1)
    xi = X.repeat(1,N)
    xj = torch.transpose(xi, 0, 1)

    out = torch.exp(- (xi - xj) ** 2. /  hyper )

    return out


# sklearn-rbf kernel takes the form k(x,y)=exp(-(x-y)**2/(2*length-scale**2))
# thus: the hyperparameter "hyper" in Gauss kernel we define equals 2 * sklearn-rbf-lengthscale**2