import pandas as pd

import matplotlib.pyplot as plt

# Define loss types and corresponding file paths
# Define loss types and corresponding file paths
loss_types = {
    'L1': 'Evaluation/Training Loss/Combined_losses.csv',  
    'L1_r6': 'Evaluation/Training Loss/Loss_L1_r6.csv',
    'L1_r8': 'Evaluation/Training Loss/Loss_L1_r8.csv'
}

# Function to load and preprocess loss data
def load_and_preprocess_loss(file_path, loss_name):
    """
    Load and preprocess loss data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        loss_name (str): Name of the loss.

    Returns:
        pd.DataFrame: Preprocessed loss data.
    """
    if loss_name == 'L1':
        # Special case: read the L1 loss from the "L1_loss" column in Combined_losses.csv
        df = pd.read_csv(file_path)
        df = df[['L1_loss']].reset_index().rename(columns={'index': 'Step', 'L1_loss': f'{loss_name}_loss'})
    else:
        # General case for other losses
        df = pd.read_csv(file_path, header=None)
        df = df.drop(0).reset_index(drop=True)
        df.columns = ['Step', f'{loss_name}_loss']
    
    df['Step'] = df['Step'].astype(int)
    df[f'{loss_name}_loss'] = df[f'{loss_name}_loss'].astype(float)
    return df

# Function to calculate mean loss per epoch (70 steps)
def calculate_mean_loss_per_epoch(df, loss_column):
    """
    Calculate the mean loss per epoch.

    Args:
        df (pd.DataFrame): Loss data.
        loss_column (str): Name of the loss column.

    Returns:
        pd.Series: Mean loss per epoch.
    """
    return df.groupby(df.index // 70)[loss_column].mean().reset_index(drop=True)

# Load and preprocess all loss data
loss_data = {loss_name: load_and_preprocess_loss(file_path, loss_name) for loss_name, file_path in loss_types.items()}

# Calculate mean loss per epoch for each loss type
mean_loss_per_epoch = {loss_name: calculate_mean_loss_per_epoch(df, f'{loss_name}_loss') for loss_name, df in loss_data.items()}


# Plot combined L1 with different ranks
plt.figure(figsize=(10, 6))
for loss_name in ['L1', 'L1_r6', 'L1_r8']:
    if loss_name in mean_loss_per_epoch:
        plt.plot(mean_loss_per_epoch[loss_name], label=loss_name, color=colors.pop(0))
plt.title('L1 Loss per Epoch with Different Ranks ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['rank4', 'rank6', 'rank8'])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Evaluation/Training Loss/Combined_L1_Ranks_Loss_per_Epoch.png')
plt.close()
