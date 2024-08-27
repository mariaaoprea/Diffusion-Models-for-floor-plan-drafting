import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = pd.read_csv('Evaluation/Training Loss/Combined_losses.csv')

# Function to calculate trend line
def add_trend_line(ax, x, y, color):
    z = np.polyfit(x, y, 2)  # Fit a polynomial of degree 2
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='dashed', color=color)

# Create the plot for the combined losses with trend lines
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Plotting L1_loss
ax1.plot(data['Epoch'], data['L1_loss'], label='L1_loss', color='red')
add_trend_line(ax1, data['Epoch'], data['L1_loss'], color='grey')
ax1.set_title('L1 Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Plotting MSE_loss
ax2.plot(data['Epoch'], data['MSE_loss'], label='MSE_loss', color='gold')
add_trend_line(ax2, data['Epoch'], data['MSE_loss'], color='grey')
ax2.set_title('MSE Loss')
ax2.set_xlabel('Epoch')
ax2.grid(True)

# Plotting SNR_loss
ax3.plot(data['Epoch'], data['SNR_loss'], label='SNR_loss', color='blue')
add_trend_line(ax3, data['Epoch'], data['SNR_loss'], color='grey')
ax3.set_title('SNR Loss')
ax3.set_xlabel('Epoch')
ax3.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('combined_loss_plot_with_trend.png')

# Show the plot
plt.show()
