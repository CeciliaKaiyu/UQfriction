from Sumatra.GPplot import *
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define paths
path = "../Data/MaxMw87"
parapath = "./sklearn_para/Mw87"
pathtest = "../Data/MaxMw87/MC_simulation"

# Load parameters and data
test_files = ["f1f.npy", "f2c.npy", "f2f.npy", "f2x1.npy", "f2x0.npy"]
path_files = {file: np.load(f"{path}/{file}") for file in test_files}

g1g2minmax = np.load(f"{parapath}/g1g2minmax.npy")
para = np.load(f"{parapath}/para.npy")
f1test = np.load(f"{pathtest}/f1f.npy")
f2test = np.load(f"{pathtest}/f2x1.npy")
xtest = np.load("../Data/MaxMw92/MC_simulation/x1.npy")

# Normalize coordinates
x2 = (np.round(np.linspace(0.01, 0.09, 5), 4)[[1, 2, 3]] - 0.01) / 0.08
x1 = (np.round(np.linspace(0.01, 0.09, 10), 4)[1:10:2] - 0.01) / 0.08
xtest = (xtest - 0.01) / 0.08
xtest = xtest[[0, 1, 2, 5, 6, 9]]
x1all = (np.round(np.linspace(0.01, 0.09, 10), 4) - 0.01) / 0.08

# Convert to tensors
g2 = torch.tensor((path_files["f2f.npy"] - path_files["f2c.npy"])[[1, 2, 3]])
g1 = torch.tensor(path_files["f1f.npy"])[1:10:2]
g2test = torch.tensor(f2test - f1test)[[0, 1, 2, 5, 6, 9], :]
g1test = torch.tensor(f1test)[[0, 1, 2, 5, 6, 9], :]
g1all = torch.tensor(path_files["f1f.npy"])
g2all = torch.tensor((path_files["f2x1.npy"] - path_files["f1f.npy"]))

# Min-max scaling function
def min_max(Y, min_Y, max_Y):
    return (Y - min_Y) / (max_Y - min_Y)

# Normalize data
g1test = min_max(g1test, g1g2minmax[0], g1g2minmax[2])
g2test = min_max(g2test, g1g2minmax[1], g1g2minmax[3])
g1 = min_max(g1, g1g2minmax[0], g1g2minmax[2])
g2 = min_max(g2, g1g2minmax[1], g1g2minmax[3])

# Define index and plotting loop
# Index list
index = [9]
for i in index:
    plt.figure(dpi=600, figsize=(8, 5))
    plt.rcParams['font.size'] = '17'
    GP_plot(xtest=xtest, X=torch.tensor(x1), Y=g1[:, i], hyper=para[0, i], ytest=g1test[:, i], level=0, min_Y=g1g2minmax[0], max_Y=g1g2minmax[2])
    plt.title(f"Gauge {i} Mw8.7")
    plt.savefig(f'./GP/GP87_coarse_{i+1}.png', bbox_inches='tight')
    plt.close()

    plt.figure(dpi=600, figsize=(8, 5))
    plt.rcParams['font.size'] = '17'
    GP_plot(xtest=xtest, X=torch.tensor(x2), Y=g2[:, i], hyper=para[2, i], ytest=g2test[:, i], level=1, min_Y=g1g2minmax[1], max_Y=g1g2minmax[3])
    plt.title(f"Gauge {i} Mw8.7")
    plt.savefig(f'./GP/GP87_fine_{i+1}.png', bbox_inches='tight')
    plt.close()
