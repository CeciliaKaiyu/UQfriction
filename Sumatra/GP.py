# -------------------
# Bayesian Quadrature (BQ) with Gaussian Process (GP)
# -------------------



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Avoid potential conflicts with MKL libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# -------------------
# Path Configurations
# -------------------
xmin, xmax = 0.01, 0.09
path = "../Data/MaxMw92"
pathsave = "./sklearn_para/Mw92"
pathtest = "../Data/MaxMw92/MC_simulation"

# Ensure the save path exists
os.makedirs(pathsave, exist_ok=True)

# -------------------
# Load Data
# -------------------
# Function evaluations at different levels
f1f = np.load(f"{path}/f1f.npy")  # f1(x1)
f2c = np.load(f"{path}/f2c.npy")  # f1(x2)
f2f = np.load(f"{path}/f2f.npy")  # f2(x2)
f2x1 = np.load(f"{path}/f2x1.npy")
f2x0 = np.load(f"{path}/f2x0.npy")

# Test data
f2test = np.load(f"{pathtest}/f2x1.npy")
f1test = np.load(f"{pathtest}/f1f.npy")

# -------------------
# Normalize Input X
# -------------------
x2 = (np.round(np.linspace(0.01, 0.09, 5), 4)[[1, 2, 3]] - xmin) / (xmax - xmin)
x1 = (np.round(np.linspace(0.01, 0.09, 10), 4)[1:10:2] - xmin) / (xmax - xmin)

xtest = np.load(f"../Data/MaxMw92/MC_simulation/x1.npy")
xtest = (xtest - xmin) / (xmax - xmin)
xtest = xtest[[0, 1, 2, 5, 6, 9]]

x1all = (np.round(np.linspace(0.01, 0.09, 10), 4) - xmin) / (xmax - xmin)

# -------------------
# Convert Function Values to Torch Tensors
# -------------------
g2 = torch.tensor(f2f - f2c)[[1, 2, 3]]
g1 = torch.tensor(f1f)[1:10:2]
g2test = torch.tensor(f2test - f1test)[[0, 1, 2, 5, 6, 9], :]
g1test = torch.tensor(f1test)[[0, 1, 2, 5, 6, 9], :]
g1all = torch.tensor(f1f)
g2all = torch.tensor(f2x1 - f1f)
g1g2minmax = np.array([torch.min(g1), torch.min(g2), torch.max(g1), torch.max(g2)])

# -------------------
# Min-Max Normalization Function
# -------------------
def min_max(Y, Y_ref):
    """
    Perform Min-Max Scaling: Normalize Y using min and max of Y_ref.

    :param Y: Tensor to be normalized
    :param Y_ref: Reference tensor to determine min/max values
    :return: Normalized tensor
    """
    min_Y, max_Y = torch.min(Y_ref), torch.max(Y_ref)
    return (Y - min_Y) / (max_Y - min_Y)


# Apply normalization
g1test = min_max(g1test, g1)
g2test = min_max(g2test, g2)
g1all = min_max(g1all, g1)
g2all = min_max(g2all, g2)
g1 = min_max(g1, g1)
g2 = min_max(g2, g2)

# -------------------
# Initialize Parameter Storage
# -------------------
para = np.zeros((4, 12))


# -------------------
# Gaussian Process Regression
# -------------------
# Index list
index = range(12)

def train_gp(x_train, y_train, x_test, y_test, x_pred, param_index, level, j):
    """
    Train Gaussian Process and plot predictions.

    :param x_train: Training input (1D array)
    :param y_train: Training output (1D tensor)
    :param x_test: Test input (1D array)
    :param y_test: Test output (1D tensor)
    :param x_pred: Prediction input (1D array)
    :param param_index: Index for parameter storage
    :param level: "coarse" or "fine" for different data resolutions
    :param j: Gauge index
    """
    # Define RBF kernel
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    # Create and train Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=30, alpha=1e-5)
    gp.fit(x_train.reshape(-1, 1), y_train.numpy())

    # Store optimized kernel parameter
    optimized_kernel = gp.kernel_
    print(f"Optimized kernel parameters {level}:", optimized_kernel)
    para[param_index, i] = 2 * (optimized_kernel.length_scale ** 2)



# Iterate over selected indices
j = 0
for i in index:
    j += 1

    # Train and plot for coarse level
    train_gp(x1, g1[:, i], xtest, g1test[:, i], np.linspace(0, 1, 100).reshape(-1, 1), param_index=0,
                      level="coarse", j=j)

    # Train and plot for fine level
    train_gp(x2, g2[:, i], xtest, g2test[:, i], np.linspace(0, 1, 100).reshape(-1, 1), param_index=2,
                      level="fine", j=j)

# -------------------
# Save Results
# -------------------
np.save(f"{pathsave}/para.npy", para)  # Save optimized parameters
np.save(f"{pathsave}/g1g2minmax.npy", g1g2minmax)  # Save min-max values
