import pandas as pd
import numpy as np
import os
from pandasgui import show
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
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

def sae_loss(y_true, y_pred):
    return tf.abs(tf.reduce_sum(y_true) - tf.reduce_sum(y_pred)) / tf.reduce_sum(y_true)

def split_time_series(df, gap_threshold, window_size):
    """ Splits the time series data into continuous segments based on the gap threshold. """
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
    df['segment'] = (df['time_diff'] > gap_threshold).cumsum()  # Assigns segment IDs
    segments = {seg_id: group.drop(columns=['time_diff', 'segment']) for seg_id, group in df.groupby('segment')}
    return {seg_id: seg for seg_id, seg in segments.items() if len(seg) >= window_size}

def create_sequences(feature_series, target_series, window_size, step_size=1):
    X_seq, y_seq = [], []
    # Here we create sequences so that each X sample is a window of electricity data
    # and y is the target at the time step immediately after the window.
    for i in range(0, len(feature_series) - window_size, step_size):
        X_seq.append(feature_series[i:i+window_size])
        y_seq.append(target_series[i+window_size//2+1])
    return np.array(X_seq), np.array(y_seq)

def concatenate_time_series(df_dict, appliance_name):
    """
    Concatenates time series DataFrames from a dictionary, shifting each subsequent DataFrame by a fixed number of years.
    
    - The first DataFrame keeps its original timestamps.
    - The second DataFrame gets shifted by +10 years.
    - The third DataFrame gets shifted by +20 years, and so on.

    :param df_dict: Dictionary where keys are names and values are DataFrames with datetime indices.
    :param appliance_name: Name of the appliance column to include in the concatenated DataFrame.
    :return: Concatenated DataFrame with adjusted datetime indices.
    """
    concatenated_df = pd.DataFrame()  # Empty DataFrame to store results
    shift_years = 0  # Start with no shift for the first DataFrame

    for idx, (df_name, df) in enumerate(df_dict.items()):
        # Ensure the DataFrame contains both 'aggregate' and the selected appliance column
        if 'aggregate' in df.columns and appliance_name in df.columns:
            # Select only the required columns
            df_selected = df[['aggregate', appliance_name]].copy()

            # Ensure index is a datetime index
            df_selected.index = pd.to_datetime(df_selected.index)

            # Apply the year shift (second DataFrame +10 years, third +20 years, etc.)
            if idx > 0:
                shift_years += 10  # Increase shift by 10 years for each subsequent DataFrame
                df_selected.index = df_selected.index + pd.DateOffset(years=shift_years)

            # Append to the concatenated DataFrame
            concatenated_df = pd.concat([concatenated_df, df_selected])

    return concatenated_df

def load_data(root_folder):
    houses = ["house_1", "house_2", "house_3", "house_4", "house_5"]
    #data_types = ["resampled_data", "normalized_data", "activation_data"]
    data_types = ["normalized_data", "resampled_data"]
    
    dataframes = {}
    for house in houses:
        for data_type in data_types:
            file_name = root_folder + f"\{house}\{data_type}.csv"
            if os.path.exists(file_name):
                df = pd.read_csv(file_name, index_col=0, parse_dates=True)
                dataframes[f"{house}_{data_type}"] = df   
    return dataframes

def make_sliding_windows(df_concat, appliance, window_size, gap_threshold, step_size):
    
    segments = split_time_series(df_concat, gap_threshold, window_size)

    X_all, y_all = [], []
    for seg_id, segment_df in segments.items():
        X_series = segment_df['aggregate'].values
        y_series = segment_df[appliance].values
        
        # Create sequences only within each continuous segment
        X_seg, y_seg = create_sequences(X_series, y_series, window_size, step_size)
        
        if X_seg.shape[0] > 0:
            X_all.append(X_seg)
            y_all.append(y_seg)
    
    return X_all, y_all

def create_min_max(df_dict):
    """
    Takes a dictionary of DataFrames and returns a summary DataFrame
    with min and max values for each column in each DataFrame,
    plus a 'total' row calculated from min of *_min and max of *_max columns.
    
    Parameters:
    - df_dict: dict, keys are identifiers (e.g. house names), values are pandas DataFrames
    
    Returns:
    - summary_df: DataFrame with one row per key in df_dict, plus a 'total' row
    """
    summary_rows = []

    for house_id, df in df_dict.items():
        row = {}
        for col in df.columns:
            row[f"{col}_min"] = df[col].min()
            row[f"{col}_max"] = df[col].max()
        row["house"] = house_id
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("house")

    # Calculate total row by taking min of *_min and max of *_max columns
    total_row = {}
    for col in summary_df.columns:
        if col.endswith("_min"):
            total_row[col] = summary_df[col].min()
        elif col.endswith("_max"):
            total_row[col] = summary_df[col].max()
    total_row_series = pd.Series(total_row, name="total_min_max")
    
    # Calculate total row by taking min of *_min and max of *_max columns
    total_mean_row = {}
    for col in summary_df.columns:
        if col.endswith("_min"):
            total_mean_row[col] = summary_df[col].mean()
        elif col.endswith("_max"):
            total_mean_row[col] = summary_df[col].mean()
    total_mean_row_series = pd.Series(total_mean_row, name="total_mean")

    # Append total row
    summary_df = pd.concat([summary_df, total_row_series.to_frame().T, total_mean_row_series.to_frame().T])

    return summary_df


def create_mean_std(df_dict):
    """
    Takes a dictionary of DataFrames and returns a summary DataFrame
    with min and max values for each column in each DataFrame,
    plus a 'total' row calculated from min of *_min and max of *_max columns.
    
    Parameters:
    - df_dict: dict, keys are identifiers (e.g. house names), values are pandas DataFrames
    
    Returns:
    - summary_df: DataFrame with one row per key in df_dict, plus a 'total' row
    """
    summary_rows = []

    for house_id, df in df_dict.items():
        row = {}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        row["house"] = house_id
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("house")
    
    # Calculate total row by taking min of *_min and max of *_max columns
    total_mean_row = {}
    for col in summary_df.columns:
        if col.endswith("_mean"):
            total_mean_row[col] = summary_df[col].mean()
        elif col.endswith("_std"):
            total_mean_row[col] = summary_df[col].mean()
    total_mean_row_series = pd.Series(total_mean_row, name="total_mean")

    # Append total row
    summary_df = pd.concat([summary_df, total_mean_row_series.to_frame().T])

    return summary_df







# ------------------------------
# loading the CSV files
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "..", "..", "data")
root_folder = os.path.abspath(data_folder)
#root_folder = r"C:\Moein\TRANSFER\data"
dataframes = load_data(root_folder)
#show(dataframes["house_1_normalized_data"].head(500))
#show(dataframes["house_1_activation_data"].head(500))
            
# ------------------------------
# Parameters & settings
# ------------------------------
loss_fun = "MAE"
normal = "std_ratio"   # std_ratio, std_std, min_max_min_max
dense_layer_size = 256
batch_size = 128
lr = 0.001
l2_reg = 0
droup_out = 0
gap_threshold = 100   # Define gap threshold in seconds
window_size = 45      # number of time steps per input sample (4200 sec)
step_size = 1         # sliding window step
columns = ["kettle", ]
for appliance in columns:

    # ------------------------------
    # Saving min and max or std and mean for the appliance
    # ------------------------------    
    df_dict = {"house_1" : dataframes["house_1_resampled_data"], 
            "house_3" : dataframes["house_3_resampled_data"],  
            "house_4" : dataframes["house_4_resampled_data"], 
            "house_5" : dataframes["house_5_resampled_data"],}
    
    if normal == "min_max_min_max":
        min_max_df = create_min_max(df_dict)
        min_max_df.to_csv("min_max_summary.csv")
    
    elif normal == "std_std":
        mean_std_df = create_mean_std(df_dict)
        mean_std_df.to_csv("mean_std_summary.csv")
    
    # ------------------------------
    # Create Sliding Window Sequences
    # ------------------------------    
    #df_train = dataframes["house_1_normalized_data"]
    df_dict = {"house_1" : dataframes["house_1_normalized_data"], 
            "house_3" : dataframes["house_3_normalized_data"],  
            "house_4" : dataframes["house_4_normalized_data"], 
            "house_5" : dataframes["house_5_normalized_data"],}
    df_concat = concatenate_time_series(df_dict, appliance)
    X_all, y_all = make_sliding_windows(df_concat, appliance, window_size, gap_threshold, step_size)
    X = np.vstack(X_all)  # Stack arrays along axis 0
    y = np.concatenate(y_all).flatten()
    # Reshape X to add the channel dimension (samples, time_steps, channels)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------
    # Build the CNN Model
    # ------------------------------
    model = Sequential()

    if l2_reg==0:
        model.add(Conv1D(filters=30, kernel_size=5,strides=1, activation='relu', input_shape=(window_size, 1)))
    else:
        model.add(Conv1D(filters=30, kernel_size=5,strides=1, activation='relu', input_shape=(window_size, 1), kernel_regularizer=l2(l2_reg)))    
    model.add(Conv1D(filters=30, kernel_size=4,strides=1, activation='relu'))
    model.add(Conv1D(filters=40, kernel_size=3,strides=1, activation='relu'))
    model.add(Conv1D(filters=50, kernel_size=3,strides=1, activation='relu'))
    model.add(Conv1D(filters=50, kernel_size=2,strides=1, activation='relu'))

    model.add(Flatten())
    model.add(Dense(dense_layer_size, activation='relu')) 
    if droup_out!=0:
        model.add(Dropout(droup_out))
    model.add(Dense(1, activation='linear'))

    # Output Layers
    seq2seq_output = Dense(window_size, activation='linear', name="seq2seq")  # Output full sequence (seq2seq)

    optimizer = Adam(learning_rate=lr)
    if loss_fun=="MSE":
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    elif loss_fun=="MAE":
        model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
        
    #model.summary()


    # ------------------------------
    # Evaluate the Model
    # ------------------------------
    # Define EarlyStopping Callback
    early_stopping = EarlyStopping(
        monitor='val_loss',   # Monitor validation loss
        patience=5,           # Stop if no improvement for 5 epochs
        restore_best_weights=True  # Keeps the best model
    )

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=batch_size,  #512 fastest but max memory
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1  # Ensures output is printed
    )


    # ------------------------------
    # Visualize and Save the Model
    # ------------------------------
    if loss_fun=="MSE":
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mae = history.history['mae']
        val_mae = history.history['val_mae']

        best_epoch = val_loss.index(min(val_loss)) + 1  # +1 because epochs start at 1
        print(f"Best Model Found at Epoch: {best_epoch}")


        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Over Epochs, {appliance}")

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(train_mae, label='Training MAE')
        plt.plot(val_mae, label='Validation MAE')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.legend()
        plt.title(f"MAE Over Epochs, {appliance}")

        plt.show()


    elif loss_fun=="MAE":
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mse = history.history['mse']
        val_mse = history.history['val_mse']

        best_epoch = val_loss.index(min(val_loss)) + 1  # +1 because epochs start at 1
        print(f"Best Model Found at Epoch: {best_epoch}")

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Over Epochs, {appliance}")

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(train_mse, label='Training MSE')
        plt.plot(val_mse, label='Validation MSE')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Square Error (MSE)")
        plt.legend()
        plt.title(f"MSE Over Epochs, {appliance}")

        plt.show()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    keras_filename = f"{script_name}.keras"
    model.save(keras_filename)
    #model.save("best_model_manual.keras")


    # ------------------------------
    # load the Model
    # ------------------------------
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    keras_filename = f"{script_name}.keras"
    #model = load_model(keras_filename)
    model.summary()

    if normal == "min_max_ratio":
        # ------------------------------
        # visualize the Model result
        # ------------------------------
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

        df_test = dataframes["house_2_normalized_data"]
        df_test_non_normalized = dataframes["house_2_resampled_data"]
        min_test = df_test_non_normalized["aggregate"].min()
        max_test = df_test_non_normalized["aggregate"].max()
        start_time =start_time_dict[appliance]
        duration = duration_dict[appliance]
        end_time = start_time + pd.Timedelta(hours=duration)

        test_segment_df = df_test.loc[start_time:end_time]

        X_all, y_all = make_sliding_windows(test_segment_df, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()
        y_pred = model.predict(X_test).flatten()

        #denormalization
        X_test_denormalized = X_test*(max_test-min_test)+min_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*X_test_denormalized_midpoint
        y_pred_denormalized = y_pred*X_test_denormalized_midpoint

        plt.figure(figsize=(12, 5))
        plt.plot(y_test_denormalized, label="Ground Truth", color="blue", linestyle="-")
        plt.plot(y_pred_denormalized, label="Predictions", color="red", linestyle="dashed")
        plt.xlabel("Time Steps")
        plt.ylabel("appliance Power (W)")
        plt.title(f"Predictions vs. Ground Truth, {appliance}")
        plt.legend()
        plt.show()

        # ------------------------------
        # calculate metrics
        # ------------------------------
        df_test = dataframes["house_2_normalized_data"]
        X_all, y_all = make_sliding_windows(df_test, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()

        y_pred = model.predict(X_test).flatten()

        # denormalization
        X_test_denormalized = X_test*(max_test-min_test)+min_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*X_test_denormalized_midpoint
        y_pred_denormalized = y_pred*X_test_denormalized_midpoint

        # Compute error metrics
        sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized)  # Signal Aggregate Error (SAE)
        mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
        mse = mean_squared_error(y_test_denormalized, y_pred_denormalized)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_denormalized, y_pred_denormalized)

        # Compile results into a dictionary
        metrics_results = {
            "SAE (Signal Aggregate Error)": sae,
            "MAE (Mean Absolute Error)": mae,
            "MSE (Mean Squared Error)": mse,
            "RMSE (Root Mean Squared Error)": rmse,
            "R² Score": r2
        }

        # Convert to DataFrame for better visualization
        df_metrics = pd.DataFrame(metrics_results, index=["Test Dataset"])
        #show(df_metrics)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        csv_filename = f"{script_name}.csv"
        df_metrics.to_csv(csv_filename, index=False)
    
    elif normal == "std_ratio":
        # ------------------------------
        # visualize the Model result
        # ------------------------------
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

        df_test = dataframes["house_2_normalized_data"]
        df_test_non_normalized = dataframes["house_2_resampled_data"]
        mean_test = df_test_non_normalized["aggregate"].mean()
        std_test = df_test_non_normalized["aggregate"].std()
        start_time =start_time_dict[appliance]
        duration = duration_dict[appliance]
        end_time = start_time + pd.Timedelta(hours=duration)

        test_segment_df = df_test.loc[start_time:end_time]

        X_all, y_all = make_sliding_windows(test_segment_df, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()
        y_pred = model.predict(X_test).flatten()

        #denormalization
        X_test_denormalized = X_test*std_test+mean_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*X_test_denormalized_midpoint
        y_pred_denormalized = y_pred*X_test_denormalized_midpoint

        plt.figure(figsize=(12, 5))
        plt.plot(y_test_denormalized, label="Ground Truth", color="blue", linestyle="-")
        plt.plot(y_pred_denormalized, label="Predictions", color="red", linestyle="dashed")
        plt.xlabel("Time Steps")
        plt.ylabel("appliance Power (W)")
        plt.title(f"Predictions vs. Ground Truth, {appliance}")
        plt.legend()
        plt.show()

        # ------------------------------
        # calculate metrics
        # ------------------------------
        df_test = dataframes["house_2_normalized_data"]
        X_all, y_all = make_sliding_windows(df_test, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()

        y_pred = model.predict(X_test).flatten()

        # denormalization
        X_test_denormalized = X_test*std_test+mean_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*X_test_denormalized_midpoint
        y_pred_denormalized = y_pred*X_test_denormalized_midpoint

        # Compute error metrics
        sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized)  # Signal Aggregate Error (SAE)
        mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
        mse = mean_squared_error(y_test_denormalized, y_pred_denormalized)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_denormalized, y_pred_denormalized)

        # Compile results into a dictionary
        metrics_results = {
            "SAE (Signal Aggregate Error)": sae,
            "MAE (Mean Absolute Error)": mae,
            "MSE (Mean Squared Error)": mse,
            "RMSE (Root Mean Squared Error)": rmse,
            "R² Score": r2
        }

        # Convert to DataFrame for better visualization
        df_metrics = pd.DataFrame(metrics_results, index=["Test Dataset"])
        #show(df_metrics)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        csv_filename = f"{script_name}.csv"
        df_metrics.to_csv(csv_filename, index=False)
        
    elif normal == "std_std":
        # ------------------------------
        # visualize the Model result
        # ------------------------------
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

        df_test = dataframes["house_2_normalized_data"]
        df_test_non_normalized = dataframes["house_2_resampled_data"]
        mean_test = df_test_non_normalized["aggregate"].mean()
        std_test = df_test_non_normalized["aggregate"].std()
        mean_test_y = df_test_non_normalized[appliance].mean()
        std_test_y = df_test_non_normalized[appliance].std()
        start_time =start_time_dict[appliance]
        duration = duration_dict[appliance]
        end_time = start_time + pd.Timedelta(hours=duration)

        test_segment_df = df_test.loc[start_time:end_time]

        X_all, y_all = make_sliding_windows(test_segment_df, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()
        y_pred = model.predict(X_test).flatten()
        
        mean_appliance_total = mean_std_df.loc["total_mean", appliance + "_mean"]
        std_appliance_total = mean_std_df.loc["total_mean", appliance + "_std"]

        #denormalization
        X_test_denormalized = X_test*std_test+mean_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*std_test_y+mean_test_y
        y_pred_denormalized = y_pred*std_appliance_total+mean_appliance_total

        plt.figure(figsize=(12, 5))
        plt.plot(y_test_denormalized, label="Ground Truth", color="blue", linestyle="-")
        plt.plot(y_pred_denormalized, label="Predictions", color="red", linestyle="dashed")
        plt.xlabel("Time Steps")
        plt.ylabel("appliance Power (W)")
        plt.title(f"Predictions vs. Ground Truth, {appliance}")
        plt.legend()
        plt.show()

        # ------------------------------
        # calculate metrics
        # ------------------------------
        df_test = dataframes["house_2_normalized_data"]
        X_all, y_all = make_sliding_windows(df_test, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()
        y_pred = model.predict(X_test).flatten()

        # denormalization
        X_test_denormalized = X_test*std_test+mean_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*std_test_y+mean_test_y
        y_pred_denormalized = y_pred*std_appliance_total+mean_appliance_total

        # Compute error metrics
        sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized)  # Signal Aggregate Error (SAE)
        mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
        mse = mean_squared_error(y_test_denormalized, y_pred_denormalized)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_denormalized, y_pred_denormalized)

        # Compile results into a dictionary
        metrics_results = {
            "SAE (Signal Aggregate Error)": sae,
            "MAE (Mean Absolute Error)": mae,
            "MSE (Mean Squared Error)": mse,
            "RMSE (Root Mean Squared Error)": rmse,
            "R² Score": r2
        }

        # Convert to DataFrame for better visualization
        df_metrics = pd.DataFrame(metrics_results, index=["Test Dataset"])
        #show(df_metrics)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        csv_filename = f"{script_name}.csv"
        df_metrics.to_csv(csv_filename, index=False)
    elif normal == "min_max_min_max":
        # ------------------------------
        # visualize the Model result
        # ------------------------------
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

        df_test = dataframes["house_2_normalized_data"]
        df_test_non_normalized = dataframes["house_2_resampled_data"]
        min_test = df_test_non_normalized["aggregate"].min()
        max_test = df_test_non_normalized["aggregate"].max()
        min_test_y = df_test_non_normalized[appliance].min()
        max_test_y = df_test_non_normalized[appliance].max()
        start_time =start_time_dict[appliance]
        duration = duration_dict[appliance]
        end_time = start_time + pd.Timedelta(hours=duration)

        test_segment_df = df_test.loc[start_time:end_time]

        X_all, y_all = make_sliding_windows(test_segment_df, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()
        y_pred = model.predict(X_test).flatten()
        
        min_appliance_total = min_max_df.loc["total_mean", appliance + "_min"]
        max_appliance_total = min_max_df.loc["total_mean", appliance + "_max"]

        #denormalization
        X_test_denormalized = X_test*(max_test-min_test)+min_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*(max_test_y-min_test_y)+min_test_y
        y_pred_denormalized = y_pred*(max_appliance_total-min_appliance_total)+min_appliance_total

        plt.figure(figsize=(12, 5))
        plt.plot(y_test_denormalized, label="Ground Truth", color="blue", linestyle="-")
        plt.plot(y_pred_denormalized, label="Predictions", color="red", linestyle="dashed")
        plt.xlabel("Time Steps")
        plt.ylabel("appliance Power (W)")
        plt.title(f"Predictions vs. Ground Truth, {appliance}")
        plt.legend()
        plt.show()

        # ------------------------------
        # calculate metrics
        # ------------------------------
        df_test = dataframes["house_2_normalized_data"]
        X_all, y_all = make_sliding_windows(df_test, appliance, window_size, gap_threshold, step_size)
        X_test = np.vstack(X_all)  # Stack arrays along axis 0
        y_test = np.concatenate(y_all).flatten()

        y_pred = model.predict(X_test).flatten()

        # denormalization
        X_test_denormalized = X_test*(max_test-min_test)+min_test
        X_test_denormalized_midpoint = extract_midpoints(X_test_denormalized)
        y_test_denormalized = y_test*(max_test_y-min_test_y)+min_test_y
        y_pred_denormalized = y_pred*(max_appliance_total-min_appliance_total)+min_appliance_total

        # Compute error metrics
        sae = np.abs(np.sum(y_test_denormalized) - np.sum(y_pred_denormalized)) / np.sum(y_test_denormalized)  # Signal Aggregate Error (SAE)
        mae = mean_absolute_error(y_test_denormalized, y_pred_denormalized)
        mse = mean_squared_error(y_test_denormalized, y_pred_denormalized)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_denormalized, y_pred_denormalized)

        # Compile results into a dictionary
        metrics_results = {
            "SAE (Signal Aggregate Error)": sae,
            "MAE (Mean Absolute Error)": mae,
            "MSE (Mean Squared Error)": mse,
            "RMSE (Root Mean Squared Error)": rmse,
            "R² Score": r2
        }

        # Convert to DataFrame for better visualization
        df_metrics = pd.DataFrame(metrics_results, index=["Test Dataset"])
        #show(df_metrics)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        csv_filename = f"{script_name}.csv"
        df_metrics.to_csv(csv_filename, index=False)