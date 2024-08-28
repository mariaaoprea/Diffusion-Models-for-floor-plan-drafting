import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
data = pd.read_csv('Evaluation/Training Loss/Loss_L1r4_MSE_SNR.csv')

# Function to calculate moving average
def moving_average(series, window_size):
    return series.rolling(window=window_size, min_periods=1).mean()

# Function to calculate trend line
def add_trend_line(ax, x, y, color):
    z = np.polyfit(x, y, 2)  # Fit a polynomial of degree 2
    p = np.poly1d(z)
    ax.plot(x, p(x), linestyle='dashed', color=color)

# Define window size for moving average
window_size = 5

# Calculate moving averages
data['L1_loss_avg'] = moving_average(data['L1_loss'], window_size)
data['MSE_loss_avg'] = moving_average(data['MSE_loss'], window_size)
data['SNR_loss_avg'] = moving_average(data['SNR_loss'], window_size)

# Create the plot for the combined losses with trend lines
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Plotting L1_loss moving average and trend line
ax1.plot(data['Epoch'], data['L1_loss_avg'], color='red')
add_trend_line(ax1, data['Epoch'], data['L1_loss_avg'], color='grey')
ax1.set_title('L1 Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

# Plotting MSE_loss moving average and trend line
ax2.plot(data['Epoch'], data['MSE_loss_avg'], color='gold')
add_trend_line(ax2, data['Epoch'], data['MSE_loss_avg'], color='grey')
ax2.set_title('MSE Loss')
ax2.set_xlabel('Epoch')
ax2.grid(True)
ax2.legend()

# Plotting SNR_loss moving average and trend line
ax3.plot(data['Epoch'], data['SNR_loss_avg'], color='blue')
add_trend_line(ax3, data['Epoch'], data['SNR_loss_avg'], color='grey')
ax3.set_title('SNR Loss')
ax3.set_xlabel('Epoch')
ax3.grid(True)
ax3.legend()

# Adjust layout
plt.tight_layout()

# Save the plot to a file
plt.savefig('combined_loss_plot_smoothed_with_trend_only.png')

# Show the plot
plt.show()
