import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define x-axis limits
xmin, xmax = 0.01, 0.09


# Function to load mean (E) and variance (V) from files
def get_EV(beta_folder, path):
    est = np.load(f"{beta_folder}/{path}/est.npy")
    estE = est[[0, 2], :]
    estV = est[[1, 3], :]
    return np.sum(estE, axis=0), np.sum(estV, axis=0)


# Function to compute normal probability density function (PDF)
def get_pdf(mean, variance, x):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance))


# Beta folder paths
beta_folders = ["./Beta25", "./Beta33"]

# Process only Mw92 dataset
path = "Mw92"
E25, V25 = get_EV(beta_folders[0], path)
E33, V33 = get_EV(beta_folders[1], path)

# Index list
index = range(len(E25))

for j, i in enumerate(index, start=1):
    # Define x range for the plot
    x_min = min(np.min(E25[index] - 3 * V25[index]), np.min(E33[index] - 3 * V33[index]))
    x_max = max(np.max(E25[index] + 3 * V25[index]), np.max(E33[index] + 3 * V33[index]))
    x = np.linspace(x_min, x_max, 1000)

    # Compute PDFs for both distributions
    pdf25 = get_pdf(E25[i], V25[i], x)
    pdf33 = get_pdf(E33[i], V33[i], x)

    # Create and format plot
    plt.figure(dpi=600, figsize=(6, 3.5))
    plt.rcParams['font.size'] = '17'
    plt.plot(x, pdf25, color='blue', linewidth=2.0, label="Beta(2,5)")
    plt.plot(x, pdf33, color='green', linewidth=2.0, label="Beta(3,3)")
    plt.axvline(x=E25[i], color='red', linestyle='--', linewidth=2.0)
    plt.axvline(x=E33[i], color='red', linestyle='--', linewidth=2.0)

    plt.xlabel('Integrated Maximum Momentum Flux (kg/$s^2$)')
    plt.ylabel('Density')
    plt.legend()

    # Save and close plot
    plt.savefig(f'./dist_plot/normal_distribution_plot_{j}.png', bbox_inches='tight')
    plt.close()

print("Plots for Mw92 have been saved.")
