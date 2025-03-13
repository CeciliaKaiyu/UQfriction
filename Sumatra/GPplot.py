import torch
import numpy as np
import matplotlib.pyplot as plt


# Use scaled X and scaled Y
# Rescale back Y after GP prediction
def GP(myx, xtest, X, Y, hyper):
    def Gauss_GP(xi, xj, hyper):
        """
        Compute Gaussian covariance matrix.
        :param xi: First input set
        :param xj: Second input set
        :param hyper: length-scale**2
        :return: Gaussian covariance matrix
        """
        Ni, Nj = xi.size()[0], xj.size()[0]

        Xi = xi.unsqueeze(1).repeat(1, Nj)
        Xj = xj.unsqueeze(1).repeat(1, Ni).T

        return torch.exp(- (Xi - Xj) ** 2. / hyper)

    myx_gp = torch.from_numpy(np.unique(np.append(np.append(myx, X), xtest)))
    nugget = 1e-6
    Inv_K = torch.inverse(Gauss_GP(X, X, hyper) + nugget * torch.eye(X.size(0)))

    gp_mean = Gauss_GP(myx_gp, X, hyper) @ Inv_K @ Y

    # GP posterior covariance
    gp_cov = Gauss_GP(myx_gp, myx_gp, hyper) - Gauss_GP(myx_gp, X, hyper) @ Inv_K @ Gauss_GP(X, myx_gp, hyper)

    return myx_gp, gp_mean, gp_cov.diagonal()


def GP_plot(xtest, X, Y, hyper, ytest, level, min_Y, max_Y):
    """
    Plot Gaussian Process regression results.
    :param xtest: Test data points
    :param X: Training input points
    :param Y: Training output values
    :param hyper: Hyperparameter (length scale**2)
    :param ytest: Test output values
    :param level: Level indicator for labeling
    :param min_Y: Minimum Y value for rescaling
    :param max_Y: Maximum Y value for rescaling
    """
    myx = np.linspace(0, 1, 100)
    myx_gp, gp_mean, myGPv = GP(myx, xtest, X, Y, hyper)

    gp_mean = (max_Y - min_Y) * gp_mean + min_Y
    lower_gp = gp_mean - torch.sqrt(myGPv) * (max_Y - min_Y) * 2
    upper_gp = gp_mean + torch.sqrt(myGPv) * (max_Y - min_Y) * 2

    plt.plot(myx_gp, lower_gp, color='lightblue', lw=0)
    plt.plot(myx_gp, gp_mean, color='deepskyblue', label='GP Mean', lw=3)
    plt.plot(myx_gp, upper_gp, color='lightblue', lw=0)
    plt.fill_between(myx_gp, lower_gp, upper_gp, color='lightblue', alpha=0.3, label='GP Mean \u00B1 2SD')
    plt.plot(X, (max_Y - min_Y) * Y + min_Y, color='tab:blue', lw=0, marker="o", label='Training Data', markersize=8)
    plt.plot(xtest, (max_Y - min_Y) * ytest + min_Y, color='tab:red', lw=0, marker="o", label='Testing Data',
             markersize=8)

    plt.xlabel("x", fontsize=22)
    plt.ylabel("$M^{Max}_1$(x)-$M^{Max}_0$(x)" if level == 1 else "$M^{Max}_0$(x)")
    plt.legend(loc=1)
