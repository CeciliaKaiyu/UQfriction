from Sumatra.BQ_with_hyper import *
from Sumatra.KM_IE import *
from Sumatra.Kernels import *
import numpy as np
import os
import torch

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define parameters
xmin, xmax = 0.01, 0.09
path = "../Data/MaxMw87"
pathsave = "./Beta33/Mw87"
parapath = "./sklearn_para/Mw87"
para = np.load(f"{parapath}/para.npy")

# Load data
path_files = {
    "f1f": "f1f.npy",
    "f2c": "f2c.npy",
    "f2f": "f2f.npy",
    "f2x1": "f2x1.npy",
    "f2x0": "f2x0.npy"
}

data = {key: np.load(f"{path}/{file}") for key, file in path_files.items()}

# Normalize coordinates
x2 = (np.round(np.linspace(xmin, xmax, 5), 4)[[1, 2, 3]] - xmin) / (xmax - xmin)
x1 = (np.round(np.linspace(xmin, xmax, 10), 4)[1:10:2] - xmin) / (xmax - xmin)

# Convert to Torch tensors
g2 = torch.tensor(data["f2f"] - data["f2c"])[[1, 2, 3]]
g1 = torch.tensor(data["f1f"])[1:10:2]

# Initialize estimation matrix
Est = torch.zeros((4, 12))

# Load test data
pathtest = f"{path}/MC_simulation"
xtest = np.load(f"../Data/MaxMw92/MC_simulation/x1.npy")
xtest = (xtest - xmin) / (xmax - xmin)
xtest = xtest[[0, 1, 2, 5, 6, 9]]

f2test = np.load(f"{pathtest}/f2x1.npy")
f1test = np.load(f"{pathtest}/f1f.npy")

g2test = torch.tensor(f2test - f1test)[[0, 1, 2, 5, 6, 9], :]
g1test = torch.tensor(f1test)[[0, 1, 2, 5, 6, 9], :]

g1g2minmax = np.load(f"{parapath}/g1g2minmax.npy")

# Min-max scaling function
def min_max(Y, min_Y, max_Y):
    return (Y - min_Y) / (max_Y - min_Y)

# Normalize data
g1test = min_max(g1test, g1g2minmax[0], g1g2minmax[2])
g2test = min_max(g2test, g1g2minmax[1], g1g2minmax[3])
g1 = min_max(g1, g1g2minmax[0], g1g2minmax[2])
g2 = min_max(g2, g1g2minmax[1], g1g2minmax[3])

# Compute BQ estimates
index = range(12)
for i in index:
    bq_E, bq_V = BQ(
        X=torch.tensor(x1), Y=g1[:, i], Hyper=torch.tensor(para[0, i]),
        nugget=1e-5, kernel=Gauss, KM=KM_Gauss33, IE=IE_Gauss33
    )
    Est[0, i] = bq_E * (g1g2minmax[2] - g1g2minmax[0]) + g1g2minmax[0]
    Est[1, i] = bq_V * (g1g2minmax[2] - g1g2minmax[0]) ** 2

    bq_E, bq_V = BQ(
        X=torch.tensor(x2), Y=g2[:, i], Hyper=torch.tensor(para[2, i]),
        nugget=1e-5, kernel=Gauss, KM=KM_Gauss33, IE=IE_Gauss33
    )
    Est[2, i] = bq_E * (g1g2minmax[3] - g1g2minmax[1]) + g1g2minmax[1]
    Est[3, i] = bq_V * (g1g2minmax[3] - g1g2minmax[1]) ** 2

# Save results
np.save(f"{pathsave}/est.npy", Est.numpy())
