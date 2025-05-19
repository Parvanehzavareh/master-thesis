import pandas as pd
import numpy as np
import os
from pandasgui import show
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

def get_activation_np(appliance_power_np, appliance_name, on_power_threshold, min_on_duration, min_off_duration):
    """
    Converts a numpy array of power readings into ON/OFF states with enforced minimum ON/OFF durations.
    
    Parameters:
        appliance_power_np (np.ndarray): 1D numpy array of appliance power readings.
        appliance_name (str): Name of the appliance (used for looking up thresholds).
        on_power_threshold (dict): Dict of power thresholds per appliance.
        min_on_duration (dict): Dict of min ON duration per appliance (in samples).
        min_off_duration (dict): Dict of min OFF duration per appliance (in samples).
    
    Returns:
        np.ndarray: 1D numpy array of binary ON/OFF states (0 or 1).
    """
    threshold = on_power_threshold.get(appliance_name, 0)
    min_on = min_on_duration.get(appliance_name, 1)
    min_off = min_off_duration.get(appliance_name, 1)

    # Initial binary ON/OFF state
    binary_series = (appliance_power_np > threshold).astype(int)

    # Apply duration enforcement
    return enforce_min_duration_np(binary_series, min_on=min_on, min_off=min_off)


def enforce_min_duration_np(binary_series, min_on, min_off):
    """
    Enforces that ON/OFF durations meet minimum duration thresholds in a binary NumPy array.

    Parameters:
        binary_series (np.ndarray): 1D array of 1s (ON) and 0s (OFF).
        min_on (int): Minimum consecutive 1s required to remain ON.
        min_off (int): Minimum consecutive 0s required to remain OFF.

    Returns:
        np.ndarray: Updated binary array after enforcing durations.
    """
    series = binary_series.copy()
    prev_state = series[0]
    count = 1  # include the first value

    for i in range(1, len(series)):
        if series[i] == prev_state:
            count += 1
        else:
            if prev_state == 1 and count < min_on:
                series[i - count:i] = 0
            elif prev_state == 0 and count < min_off:
                series[i - count:i] = 1
            count = 1
            prev_state = series[i]

    # Post-process final segment
    if prev_state == 1 and count < min_on:
        series[-count:] = 0
    elif prev_state == 0 and count < min_off:
        series[-count:] = 1

    return series

def classification_metrics(y_true, y_pred):
    """
    Computes standard classification metrics for binary ON/OFF predictions.

    Parameters:
        y_true (np.ndarray): Ground truth binary array (0/1).
        y_pred (np.ndarray): Predicted binary array (0/1).

    Returns:
        dict: Dictionary with accuracy, precision, recall, and F1 score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

# ------------------------------
# Activation Parameters
# ------------------------------

# Power thresholds for ON/OFF classification from me
on_power_threshold = {
    "kettle": 1500,   #1000
    "microwave": 500,  #50
    "fridge_freezer": 50,  #5
    "dishwasher": 800,   #10
    "washer_dryer": 500   #20
}

# Appliance-specific min ON/OFF durations (in number of samples)
min_on_duration = {
    "kettle": 1,   #20       # Kettle must be ON for at least 10 samples
    "microwave": 1,   #10     # Microwave must be ON for at least 5 samples
    "fridge_freezer": 10, #60 # Fridge must be ON for at least 20 samples
    "dishwasher": 5,  #1800    # Dishwasher must be ON for at least 15 samples
    "washer_dryer": 5  #1800   # Washer/Dryer must be ON for at least 30 samples
}

min_off_duration = {
    "kettle": 0,   #10        # Kettle must be OFF for at least 5 samples
    "microwave": 0,  #10      # Microwave must be OFF for at least 3 samples
    "fridge_freezer": 10, #60 # Fridge must be OFF for at least 10 samples
    "dishwasher": 5,   #300   # Dishwasher must be OFF for at least 10 samples
    "washer_dryer": 5   #300  # Washer/Dryer must be OFF for at least 15 samples
}


# ------------------------------
# loading the CSV files
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "..", "data")
root_folder = os.path.abspath(data_folder)

# ------------------------------
# Parameters & settings
# ------------------------------

normal_dict = {
    "seq2seq-kettle" : "std_ratio",
    "seq2seq-fridge_freezer" : "std_ratio",
    "seq2seq-dishwasher" : "std_ratio",
    "seq2seq-microwave" : "std_ratio",
    "seq2seq-washer_dryer" : "min_max_ratio",
    "seq2point-kettle" : "std_ratio",
    "seq2point-fridge_freezer" : "std_ratio",
    "seq2point-dishwasher" : "std_ratio",
    "seq2point-microwave" : "std_ratio",
    "seq2point-washer_dryer" : "std_ratio",
}

dt_dict = {
    "seq2seq-kettle" : "45S",
    "seq2seq-fridge_freezer" : "30S",
    "seq2seq-dishwasher" : "60S",
    "seq2seq-microwave" : "45S",
    "seq2seq-washer_dryer" : "30S",
    "seq2point-kettle" : "30S",
    "seq2point-fridge_freezer" : "60S",
    "seq2point-dishwasher" : "60S",
    "seq2point-microwave" : "30S",
    "seq2point-washer_dryer" : "45S",
}

window_size_dict = {
    "seq2seq-kettle" : 31,
    "seq2seq-fridge_freezer" : 91,
    "seq2seq-dishwasher" : 69,
    "seq2seq-microwave" : 31,
    "seq2seq-washer_dryer" : 139,
    "seq2point-kettle" : 45,
    "seq2point-fridge_freezer" : 45,
    "seq2point-dishwasher" : 69,
    "seq2point-microwave" : 45,
    "seq2point-washer_dryer" : 91,
}

gap_threshold = 100   # Define gap threshold in seconds
step_size = 1         # sliding window step
appliances = ["kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"] 


house_list = ["house_1", "house_2", "house_3", "house_4", "house_5", "house_6",
              "house_7", "house_8", "house_9", "house_10", "house_11", "house_12",
              "house_13", "house_15", "house_16", "house_17", "house_18", "house_19",
              "house_20", "house_21",]


"""
house_list = ["house_2", "house_3", "house_5", "house_6",
              "house_9", "house_11", "house_15", "house_20"]
"""


classification_results = []
regression_results = []

i = 0
for house in house_list:
    for method in ["seq2seq", "seq2point"]:
        for appliance in appliances:
            key = method + "-"+appliance
            window_size = window_size_dict[key]
            normal = normal_dict[key]
            dt = dt_dict[key]
            
            file_name_denorm = root_folder + rf"\refit\resampled_data_{normal}_{dt}_{house}.csv"
            file_name_norm = root_folder + rf"\refit\normalized_data_{normal}_{dt}_{house}.csv"
            df_test = pd.read_csv(file_name_norm, index_col=0, parse_dates=True)       
            df_test_denormalized = pd.read_csv(file_name_denorm, index_col=0, parse_dates=True) 
            if appliance not in df_test_denormalized.columns:
                continue
            mean_test = df_test_denormalized["aggregate"].mean()
            std_test = df_test_denormalized["aggregate"].std()
            min_test = df_test_denormalized["aggregate"].min()
            max_test = df_test_denormalized["aggregate"].max()
            
            test_segment_df = df_test
            df_test_segment_denormalized = df_test_denormalized
            
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
                if normal == "std_ratio":
                    X_test_denormalized = X_test_reconstructed*std_test+mean_test
                elif normal == "min_max_ratio":
                    X_test_denormalized = X_test_reconstructed*(max_test-min_test)+min_test
                y_test_denormalized = y_test_reconstructed*X_test_denormalized
                y_pred_denormalized = y_pred_reconstructed*X_test_denormalized
                y_test_activation = get_activation_np(y_test_denormalized, appliance, on_power_threshold, min_on_duration, min_off_duration)
                y_pred_activation = get_activation_np(y_pred_denormalized, appliance, on_power_threshold, min_on_duration, min_off_duration)
                class_metrics = classification_metrics(y_test_activation, y_pred_activation)
                sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized) 
                mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
                reg_metrics = {"sae" : sae, "mae" : mae}         
                
            else:
                X_all, y_all = make_sliding_windows_seq2point(test_segment_df, appliance, window_size, gap_threshold, step_size)
                X_test = np.vstack(X_all)  # Stack arrays along axis 0
                y_test = np.concatenate(y_all).flatten()
                y_pred = model.predict(X_test).flatten()
                if normal == "std_ratio":        
                    X_test_denormalized = X_test*std_test+mean_test
                elif normal == "min_max_ratio":
                    X_test_denormalized = X_test*(max_test-min_test)+min_test
                X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
                y_test_denormalized = y_test*X_test_denormalized_midpoint
                y_pred_denormalized = y_pred*X_test_denormalized_midpoint
                y_test_activation = get_activation_np(y_test_denormalized, appliance, on_power_threshold, min_on_duration, min_off_duration)
                y_pred_activation = get_activation_np(y_pred_denormalized, appliance, on_power_threshold, min_on_duration, min_off_duration)
                class_metrics = classification_metrics(y_test_activation, y_pred_activation)
                sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized) 
                mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
                reg_metrics = {"sae" : sae, "mae" : mae}
                
            classification_results.append({
                                "appliance": appliance,
                                "method": method,
                                "house": house,
                                **class_metrics})
                                
            regression_results.append({
                                "appliance": appliance,
                                "method": method,
                                "house": house,
                                **reg_metrics})
            i += 1

df_class_metrics = pd.DataFrame(classification_results)
df_class_metrics.to_csv("classification_metrics.csv", index=False)

df_reg_metrics = pd.DataFrame(regression_results)
df_reg_metrics.to_csv("regression_metrics.csv", index=False)

df_class_metrics_average = (
    df_class_metrics.drop(columns=["house"])  # Remove 'house' from averaging
      .groupby(["appliance", "method"], as_index=False)
      .mean(numeric_only=True)
)

df_reg_metrics_average = (
    df_reg_metrics.drop(columns=["house"])  # Remove 'house' from averaging
      .groupby(["appliance", "method"], as_index=False)
      .mean(numeric_only=True)
)

df_class_metrics_average.to_csv("classification_metrics_average.csv", index=False)
df_reg_metrics_average.to_csv("regression_metrics_average.csv", index=False)

show(df_class_metrics)
show(df_reg_metrics)
print("FINISHED!")
