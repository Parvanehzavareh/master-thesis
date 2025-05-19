import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

output_file = os.path.join(script_dir, "combined_output.csv")

# Delete the file if it already exists
if os.path.exists(output_file):
    os.remove(output_file)

# List all CSV files in the same folder
csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]

# Read and concatenate all CSVs
df_list = []
for file in csv_files:
    file_path = os.path.join(script_dir, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# Concatenate all rows
combined_df = pd.concat(df_list, ignore_index=True)
combined_df = combined_df.sort_values(by=["appliance", "method", "SAE"])

desired_order = ["appliance", "method", "batch size", "NN architecture", "sampling resolution", "dense layer size", "dropout", "l2 regularization", "learning rate", "normalization", "window duration", "SAE", "MAE",]
combined_df = combined_df[desired_order]

# Save to a new CSV
combined_df.to_csv(os.path.join(script_dir, "combined_output.csv"), index=False)



