import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import os

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load 0.03 case data
path = "../Data/MaxMw87"
path2f = path + "/f2f.npy"  # f2(x2)
x2 = np.round(np.linspace(0.01, 0.09, 5), 4)
f2f = np.load(path2f)

# Define index array

# Load Mw8.7 roughness 0.03 max momentum flux
f003 = f2f[1, :]

# Load MaxMw87 estimated data
est87_25 = np.load("./Beta25/Mw87/est.npy")
est87_33 = np.load("./Beta33/Mw87/est.npy")

# Extract and process estimates for Beta(2,5)
estE87_25 = est87_25[[0, 2], :]
estV87_25 = est87_25[[1, 3], :]
E_sum87_25 = np.sum(estE87_25, axis=0)
V_sum87_25 = np.sum(estV87_25, axis=0)

# Extract and process estimates for Beta(3,3)
estE87_33 = est87_33[[0, 2], :]
estV87_33 = est87_33[[1, 3], :]
E_sum87_33 = np.sum(estE87_33, axis=0)
V_sum87_33 = np.sum(estV87_33, axis=0)

# Define gauge labels
classes = ['Gauge 1', 'Gauge 2', 'Gauge 3', 'Gauge 4', 'Gauge 5', 'Gauge 6',
           'Gauge 7', 'Gauge 8', 'Gauge 9', 'Gauge 10', 'Gauge 11', 'Gauge 12']

# Set up figure and axis
fig, ax = plt.subplots(dpi=300)
ind = np.arange(len(classes))  # X locations for the groups
bar_width = 0.25  # Width of the bars

# Create bars for the three categories
ref_bars = ax.bar(ind, f003, bar_width, label='n=0.03')
beta25_bars = ax.bar(ind + bar_width, E_sum87_25, bar_width, label='n Integrated Beta(2,5)')
beta33_bars = ax.bar(ind + 2 * bar_width, E_sum87_33, bar_width, label='n Integrated Beta(3,3)')

# Add labels, legend, and formatting
ax.set_xlabel('Gauges')
ax.set_ylabel('Maximum Momentum Flux (kg/$s^2$)')
ax.set_xticks(ind + bar_width)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.legend()

# Save and show plot
plt.tight_layout()
plt.savefig('./barplot87.png')
plt.show()
