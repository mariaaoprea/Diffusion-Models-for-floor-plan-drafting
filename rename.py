import os

# Define the old and new labels
old_labels = ["BFMBM", "BFOMB", "BMFBM", "BMOMB", "SFMBM", "SFOMB", "SFOMF", "SFOSM", "SFOSF"]
new_labels = ["BFMBM", "BFOBM", "BMMBM", "BMOBM", "SFMBM", "SFOBM", "SFOBF", "SFOSM", "SFOSF"]

# Loop through each old label and its corresponding new label
for label_old, label_new in zip(old_labels, new_labels):
    for i in range(1, 11):
        # Define the old file name and the new file name
        old_file = f"images/L1/{label_old}_{i}.jpg"
        new_file = f"images/L1/{label_new}_{i}.jpg"
        
        # Rename the file
        os.rename(old_file, new_file)
print("Files renamed successfully!")