import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the Beta distribution
a, b = 2, 5  # Example parameters for the beta distribution
c, d = 3, 3

# Generate x values
x = np.linspace(0, 1, 1000)
r = 0.08 * x + 0.01

# Compute the Beta PDF
y_ab = beta.pdf(x, a, b)
y_cd = beta.pdf(x, c, d)

# Calculate the mode of the Beta distribution
mode_ab = (a - 1) / (a + b - 2)
r_mode_ab = 0.08 * mode_ab + 0.01
mode_cd = (c - 1) / (c + d - 2)
r_mode_cd = 0.08 * mode_cd + 0.01

# Create the plot
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 15})
plt.plot(r, y_ab, label=f'Beta({a}, {b})', color='b')
plt.plot(r, y_cd, label=f'Beta({c}, {d})', color='g')
plt.axvline(r_mode_ab, color='orange', linestyle='--')
plt.axvline(r_mode_cd, color='orange', linestyle='--')

plt.xlabel("Manning's Roughness Coefficient n")
plt.ylabel('Probability Density')
plt.xlim((0.01, 0.09))
plt.ylim((0, 2.5))
plt.legend()

# Remove the grid
plt.grid(False)

# Save the plot
plt.savefig('beta_pdf_plot.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()