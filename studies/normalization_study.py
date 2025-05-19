import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the folders (each corresponds to a batch size)
folders = [
    "CNN - dt=60 - dense=256 - batchsize=1024",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_ratio",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=std_std",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_min_max",
]

# Define the appliances and methods
appliances = ["dishwasher", "fridge_freezer", "kettle", "microwave", "washer_dryer"]
methods = ["seq2seq", "seq2point"]

# Error metrics to plot
error_metrics = ["SAE (Signal Aggregate Error)", "MAE (Mean Absolute Error)", 
                 "MSE (Mean Squared Error)", "RMSE (Root Mean Squared Error)", "R² Score"]

# Extract batch sizes from folder names
folder1 = {
    "CNN - dt=60 - dense=256 - batchsize=1024" : "CNN - dt=60 - dense=256 - batchsize=1024 - normal=std_ratio",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_ratio" : "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_ratio",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=std_std" : "CNN - dt=60 - dense=256 - batchsize=1024 - normal=std_std",
    "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_min_max" : "CNN - dt=60 - dense=256 - batchsize=1024 - normal=min_max_min_max",
}
normalizations = [folder1[folder].split("normal=")[1] for folder in folders]

# Initialize a dictionary to store data
data_dict = {appliance: {method: {normal: None for normal in normalizations} for method in methods} for appliance in appliances}

# Parse CSV files from each folder
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Go one folder back
for folder in folders:
    normal = folder1[folder].split("normal=")[1]  # Extract batch size
    for file in os.listdir(os.path.join(parent_dir, folder)):
        if file.endswith(".csv") and file!="comparison_metrics.csv" and file!='min_max_summary.csv': 
            # Extract method (seq2seq/seq2point) and appliance from filename
            parts = file.replace("CNN-", "").replace(".csv", "").split("-")
            method, appliance = parts[0], parts[1]

            # Read the CSV file
            df = pd.read_csv(os.path.join(parent_dir, folder, file))

            # Store the dataframe in the dictionary
            if appliance in appliances and method in methods:
                data_dict[appliance][method][normal] = df

# ------------------------------
# Plotting
# ------------------------------
results = []
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), constrained_layout=True)
for i, appliance in enumerate(appliances):
    for j, method in enumerate(methods):
        ax = axes[i, j]

        # Extract data for this appliance-method pair
        sae_values = []
        mae_values = []
        for normal in normalizations:
            df = data_dict[appliance][method][normal]
            sae = df["SAE (Signal Aggregate Error)"].values[0]
            mae = df["MAE (Mean Absolute Error)"].values[0]
            sae_values.append(sae)
            mae_values.append(mae)
            results.append({
                "appliance": appliance,
                "method": method,
                "normalization": normal,
                "SAE": sae,
                "MAE": mae,
                "NN architecture" : "CNN",
                "sampling resolution" : "60s",
                "batch size" : 1024,
                "dense layer size": 256,
                "dropout": 0,
                "l2 regularization": 0,
                "learning rate": 0.001,
                "window duration": '69x60',
            })
            
        # Convert batch sizes to sorted indices
        #batch_indices = sorted(batch_sizes)
        batch_indices = normalizations

        # Primary axis (MAE)
        ax.set_xlabel("normalization")
        ax.set_ylabel("MAE")
        ax.plot(batch_indices, mae_values, marker='o', linestyle='-', color='blue', label="MAE")

        # Secondary axis (SAE)
        ax2 = ax.twinx()
        ax2.set_ylabel("SAE")
        ax2.plot(batch_indices, sae_values, marker='s', linestyle='--', color='red', label="SAE")

        # Titles and labels
        ax.set_title(f"{method.upper()} - {appliance.capitalize()}")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add legends
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

# Add a main title
fig.suptitle("Comparison of SAE & MAE Across Different normalizations", fontsize=16)

# Show the plot
plt.show()

# ------------------------------
# Save results as CSV
# ------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["appliance", "method", "normalization"])

script_name = os.path.splitext(os.path.basename(__file__))[0]
csv_filename = f"{script_name}.csv"
results_df.to_csv(csv_filename, index=False)