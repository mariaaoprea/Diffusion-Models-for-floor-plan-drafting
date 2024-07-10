import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
loss_l1_4 = pd.read_csv('losses/Loss_L1.csv', header=None)
loss_l1_6 = pd.read_csv('losses/Loss_L1_6.csv', header=None)
loss_l1_8 = pd.read_csv('losses/Loss_L1_8.csv', header=None)

# Remove the header row and convert columns to appropriate data types
loss_l1_6 = loss_l1_6.drop(0).reset_index(drop=True)
loss_l1_8 = loss_l1_8.drop(0).reset_index(drop=True)
loss_l1_4 = loss_l1_4.drop(0).reset_index(drop=True)

loss_l1_6.columns = ['Step', 'L1_6_loss']
loss_l1_8.columns = ['Step', 'L1_8_loss']
loss_l1_4.columns = ['Step', 'L1_4_loss']

loss_l1_6['Step'] = loss_l1_6['Step'].astype(int)
loss_l1_6['L1_6_loss'] = loss_l1_6['L1_6_loss'].astype(float)

loss_l1_8['Step'] = loss_l1_8['Step'].astype(int)
loss_l1_8['L1_8_loss'] = loss_l1_8['L1_8_loss'].astype(float)

loss_l1_4['Step'] = loss_l1_4['Step'].astype(int)
loss_l1_4['L1_4_loss'] = loss_l1_4['L1_4_loss'].astype(float)

# Function to calculate mean loss per epoch (70 steps)
def calculate_mean_loss_per_epoch(df, loss_column):
    return df.groupby(df.index // 70)[loss_column].mean().reset_index(drop=True)

# Calculate mean loss per epoch
l1_6_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_l1_6, 'L1_6_loss')
l1_4_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_l1_4, 'L1_4_loss')
l1_8_mean_loss_per_epoch = calculate_mean_loss_per_epoch(loss_l1_8, 'L1_8_loss')

'''# Plotting L1 Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(l1_6_mean_loss_per_epoch, label='L1_6 Loss', color='blue')
plt.title('L1_6 Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/L1_6_Loss_per_Epoch.png')
plt.close()

# Plotting L1 Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(l1_8_mean_loss_per_epoch, label='L1_8 Loss', color='blue')
plt.title('L1_8 Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/L1_8_Loss_per_Epoch.png')
plt.close()'''


# Plotting Combined Loss per Epoch
plt.figure(figsize=(10, 6))
plt.plot(l1_4_mean_loss_per_epoch, label='Rank 4', color='blue')
plt.plot(l1_6_mean_loss_per_epoch, label='Rank 6', color='green')
plt.plot(l1_8_mean_loss_per_epoch, label='Rank 8', color='red')
plt.title('L1 Loss with different ranks')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('losses/L1_different_ranks.png')
plt.close()