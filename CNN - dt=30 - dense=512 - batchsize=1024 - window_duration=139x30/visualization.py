import pandas as pd
import numpy as np
import os
from pandasgui import show
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def extract_midpoints(X):
    """
    Extracts the midpoint value from each window in X.

    :param X: 2D NumPy array of shape (num_windows, window_size)
    :return: 1D NumPy array of midpoints (num_windows,)
    """
    window_size = X.shape[1]  # Get the window size
    mid_index = window_size // 2 + 1  # Find the middle index

    # Extract the middle value from each window
    midpoints = X[:, mid_index]

    return midpoints

def reconstruct_time_series(y_pred_windows, window_size, step_size=1):
    """
    Merges overlapping window-based predictions into a continuous time series.
    
    :param y_pred_windows: 2D array of shape (num_windows, window_size)
    :param window_size: Number of time steps in each window
    :param step_size: How much each window moves forward (default=1 for full overlap)
    :return: Reconstructed 1D time series
    """
    num_windows = y_pred_windows.shape[0]
    total_length = num_windows * step_size + window_size - step_size  # Approximate final time series length

    # Initialize arrays for summed values and counts
    time_series = np.zeros(total_length)
    counts = np.zeros(total_length)

    # Merge overlapping windows
    for i in range(num_windows):
        time_series[i:i+window_size] += y_pred_windows[i]  # Accumulate values
        counts[i:i+window_size] += 1  # Track how many times each position is predicted

    # Normalize by dividing accumulated sum by counts (avoiding division by zero)
    time_series = np.divide(time_series, counts, where=counts > 0)

    return time_series


def split_time_series(df, gap_threshold, window_size):
    """ Splits the time series data into continuous segments based on the gap threshold. """
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
    df['segment'] = (df['time_diff'] > gap_threshold).cumsum()  # Assigns segment IDs
    
    segments = {seg_id: group.drop(columns=['time_diff', 'segment']) for seg_id, group in df.groupby('segment')}
    return {seg_id: seg for seg_id, seg in segments.items() if len(seg) >= window_size}

def create_sequences_seq2seq(feature_series, target_series, window_size, step_size=1):
    X_seq, y_seq = [], []
    # Here we create sequences so that each X sample is a window of electricity data
    # and y is the target at the time step immediately after the window.
    for i in range(0, len(feature_series) - window_size, step_size):
        X_seq.append(feature_series[i:i+window_size])
        y_seq.append(target_series[i:i+window_size])
    return np.array(X_seq), np.array(y_seq)

def create_sequences_seq2point(feature_series, target_series, window_size, step_size=1):
    X_seq, y_seq = [], []
    # Here we create sequences so that each X sample is a window of electricity data
    # and y is the target at the time step immediately after the window.
    for i in range(0, len(feature_series) - window_size, step_size):
        X_seq.append(feature_series[i:i+window_size])
        y_seq.append(target_series[i+window_size//2+1])
    return np.array(X_seq), np.array(y_seq)

def load_data(root_folder):
    houses = ["house_1", "house_2", "house_3", "house_4", "house_5"]
    #data_types = ["resampled_data", "normalized_data", "activation_data"]
    data_types = ["normalized_data","resampled_data"]
    
    dataframes = {}
    for house in houses:
        for data_type in data_types:
            file_name = root_folder + f"\{house}\{data_type}.csv"
            if os.path.exists(file_name):
                df = pd.read_csv(file_name, index_col=0, parse_dates=True)
                dataframes[f"{house}_{data_type}"] = df   
    return dataframes

def make_sliding_windows_seq2seq(df_concat, appliance, window_size, gap_threshold, step_size):
    
    segments = split_time_series(df_concat, gap_threshold, window_size)

    X_all, y_all = [], []
    for seg_id, segment_df in segments.items():
        X_series = segment_df['aggregate'].values
        y_series = segment_df[appliance].values
        
        # Create sequences only within each continuous segment
        X_seg, y_seg = create_sequences_seq2seq(X_series, y_series, window_size, step_size)
        
        if X_seg.shape[0] > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
    
    return X_all, y_all


def make_sliding_windows_seq2point(df_concat, appliance, window_size, gap_threshold, step_size):
    
    segments = split_time_series(df_concat, gap_threshold, window_size)

    X_all, y_all = [], []
    for seg_id, segment_df in segments.items():
        X_series = segment_df['aggregate'].values
        y_series = segment_df[appliance].values
        
        # Create sequences only within each continuous segment
        X_seg, y_seg = create_sequences_seq2point(X_series, y_series, window_size, step_size)
        
        if X_seg.shape[0] > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
    
    return X_all, y_all






# ------------------------------
# loading the CSV files
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "..", "data")
root_folder = os.path.abspath(data_folder)
dataframes = load_data(root_folder)
#show(dataframes["house_1_normalized_data"].head(500))
#show(dataframes["house_1_activation_data"].head(500))

# ------------------------------
# Parameters & settings
# ------------------------------
gap_threshold = 100   # Define gap threshold in seconds
window_size = 139      # number of time steps per input sample (4200 sec)
step_size = 1         # sliding window step
appliances = ["kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"] 
methods = ["seq2seq", "seq2point"]

# ------------------------------
# making the comparison table
# ------------------------------
df_list = []
for appliance in appliances:
    for method in methods:
        filename = f"CNN-{method}-{appliance}.csv"  # Generate file name
        df = pd.read_csv(filename) 
        df.insert(0, "appliance", appliance)
        df.insert(1, "method", method)
        df_list.append(df)  
        
final_df = pd.concat(df_list, ignore_index=True)
show(final_df) 
output_filename = "comparison_metrics.csv"  
final_df.to_csv(output_filename, index=False)






# ------------------------------
# visualize the Model results
# ------------------------------
df_test = dataframes["house_2_normalized_data"]
df_test_denormalized = dataframes["house_2_resampled_data"]
mean_test = df_test_denormalized["aggregate"].mean()
std_test = df_test_denormalized["aggregate"].std()

# Generate example data for 15 curves
x = np.linspace(0, 10, 100)
y_values = [np.sin(x + i) for i in range(15)]  # 15 different sine waves

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))  # 5 rows, 3 columns
axes = axes.flatten()  # Flatten to make indexing easier

start_time_dict = {"dishwasher" : pd.to_datetime("2013-07-01 18:00:00"), 
                 "fridge_freezer" : pd.to_datetime("2013-07-01 18:00:00"), 
                 "kettle" : pd.to_datetime("2013-07-01 6:00:00"), 
                 "microwave" : pd.to_datetime("2013-07-02 5:00:00"), 
                 "washer_dryer" : pd.to_datetime("2013-07-05 21:00:00") }

duration_dict = {"dishwasher" : 8,
              "fridge_freezer" : 8,
              "kettle" : 12,
              "microwave" : 10,
              "washer_dryer" : 24
}

appliances = ["kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"] 
methods = ["ground_truth", "seq2seq", "seq2point"]

i = 0
for appliance in appliances:
    start_time = start_time_dict[appliance]
    duration = duration_dict[appliance]
    end_time = start_time + pd.Timedelta(hours=duration)
    test_segment_df = df_test.loc[start_time:end_time]
    df_test_segment_denormalized = df_test_denormalized.loc[start_time:end_time]

    
    for method in methods:
        if method != "ground_truth":
            keras_filename = f"CNN-{method}-{appliance}.keras"  # Generate file name
            model = load_model(keras_filename)
            
            if method == "seq2seq":
                X_all, y_all = make_sliding_windows_seq2seq(test_segment_df, appliance, window_size, gap_threshold, step_size)
                X_test = np.vstack(X_all)  # Stack arrays along axis 0
                y_test = np.vstack(y_all)  # Stack target arrays along axis 0
                y_pred_windows = model.predict(X_test)
                y_pred_reconstructed = reconstruct_time_series(y_pred_windows, window_size=window_size, step_size=1)
                y_test_reconstructed = reconstruct_time_series(y_test, window_size=window_size, step_size=1)
                X_test_reconstructed = reconstruct_time_series(X_test, window_size=window_size, step_size=1)
                X_test_denormalized = X_test_reconstructed*std_test+mean_test
                y_test_denormalized = y_test_reconstructed*X_test_denormalized
                y_pred_denormalized = y_pred_reconstructed*X_test_denormalized
                
            else:
                X_all, y_all = make_sliding_windows_seq2point(test_segment_df, appliance, window_size, gap_threshold, step_size)
                X_test = np.vstack(X_all)  # Stack arrays along axis 0
                y_test = np.concatenate(y_all).flatten()
                y_pred = model.predict(X_test).flatten()              
                X_test_denormalized = X_test*std_test+mean_test
                X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
                y_test_denormalized = y_test*X_test_denormalized_midpoint
                y_pred_denormalized = y_pred*X_test_denormalized_midpoint
                            
            axes[i].plot(y_test_denormalized, label="Ground Truth", color="blue", linestyle="-")
            axes[i].plot(y_pred_denormalized, label="Predictions", color="red", linestyle="dashed")
            axes[i].legend()
            axes[i].grid(True)
        else:
            X = df_test_segment_denormalized['aggregate'].values
            y = df_test_segment_denormalized[appliance].values
            axes[i].plot(X, label="Aggregate Power", color="black", linestyle="-")
            axes[i].plot(y, label="Appliance Power", color="blue", linestyle="-")
            axes[i].legend()
            axes[i].grid(True)
        
        i += 1

# Adjust layout
axes[0].set_title(f"aggregate vs. appliance")
axes[1].set_title(f"seq2seq")
axes[2].set_title(f"seq2point")
axes[0].set_ylabel(f"kettle", rotation=90, fontsize=12, labelpad=10)
axes[3].set_ylabel(f"fridge_freezer", rotation=90, fontsize=12, labelpad=10)
axes[6].set_ylabel(f"dishwasher", rotation=90, fontsize=12, labelpad=10)
axes[9].set_ylabel(f"microwave", rotation=90, fontsize=12, labelpad=10)
axes[12].set_ylabel(f"washer_dryer", rotation=90, fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

#plt.xlabel("Time Steps")
#plt.ylabel("Scaled Power (W)")
#plt.title("Predictions vs. Ground Truth")