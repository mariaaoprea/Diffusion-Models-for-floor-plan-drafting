import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
loss_l1 = pd.read_csv('losses/Loss_L1.csv', header=None)
loss_mse = pd.read_csv('losses/Loss_MSE.csv', header=None)
loss_snr = pd.read_csv('losses/Loss_SNR.csv', header=None)

# Remove the header row and convert columns to appropriate data types
loss_l1 = loss_l1.drop(0).reset_index(drop=True)
loss_mse = loss_mse.drop(0).reset_index(drop=True)
loss_snr = loss_snr.drop(0).reset_index(drop=True)

loss_l1.columns = ['Step', 'L1_loss']
loss_mse.columns = ['Step', 'MSE_loss']
loss_snr.columns = ['Step', 'SNR_loss']

loss_l1['Step'] = loss_l1['Step'].astype(int)
loss_l1['L1_loss'] = loss_l1['L1_loss'].astype(float)

loss_mse['Step'] = loss_mse['Step'].astype(int)
loss_mse['MSE_loss'] = loss_mse['MSE_loss'].astype(float)

loss_snr['Step'] = loss_snr['Step'].astype(int)
loss_snr['SNR_loss'] = loss_snr['SNR_loss'].astype(float)

# Function to calculate mean loss per epoch (70 steps)
def calculate_mean_loss_per_epoch(df, loss_column):
    return df.groupby(df.index // 70)[loss_column].mean().reset_index(drop=True)

# Calculate mean loss per epoch
l1_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_l1, 'L1_loss')
mse_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_mse, 'MSE_loss')
snr_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_snr, 'SNR_loss')

# Plotting L1 Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(l1_mean_loss_per_epoch, label='L1 Loss', color='blue')
plt.title('L1 Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/L1_Loss_per_Epoch.png')
plt.close()

# Plotting MSE Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(mse_mean_loss_per_epoch, label='MSE Loss', color='green')
plt.title('MSE Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/MSE_Loss_per_Epoch.png')
plt.close()

# Plotting SNR Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(snr_mean_loss_per_epoch, label='SNR Loss', color='red')
plt.title('SNR Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/SNR_Loss_per_Epoch.png')
plt.close()

# Plotting Combined Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(l1_mean_loss_per_epoch, label='L1 Loss', color='blue')
plt.plot(mse_mean_loss_per_epoch, label='MSE Loss', color='green')
plt.plot(snr_mean_loss_per_epoch, label='SNR Loss', color='red')
plt.title('Combined Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/Combined_Loss_per_Epoch.png')
plt.close()