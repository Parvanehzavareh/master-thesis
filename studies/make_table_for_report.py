import pandas as pd
from pandasgui import show

dev_flag = False


file_path = "window_duration_study.csv"
parameter="window duration"
window_durations = ["23x60", "45x60", "69x60"]
base_window = "69x60"  


file_path = "batchsize_study.csv"
parameter="batch size"
window_durations = ["128", "256", "1024"]
base_window = "1024"  

file_path = "dense_layer_size_study.csv"
parameter="dense layer size"
window_durations = ["128", "256", "256,128", "512"]
base_window = "256"  

file_path = "dropout_study.csv"
parameter="dropout"
window_durations = ["0.0", "0.25", "0.5"]
base_window = "0.0"  

file_path = "l2_reg_study.csv"
parameter="l2 regularization"
window_durations = ["0.0", "5e-05", "0.0001", "0.0002", "0.001"]
base_window = "0.0"  

file_path = "learning_rate_study.csv"
parameter="learning rate"
window_durations = ["0.0001", "0.0005", "0.001", "0.005"]
base_window = "0.001"

file_path = "normalization_study.csv"
parameter="normalization"
window_durations = ["min_max_min_max", "min_max_ratio", "std_ratio", "std_std"]
base_window = "std_ratio"
"""
file_path = "sampling_resolution_study.csv"
parameter="sampling resolution"
window_durations = ["30S", "45S", "60S"]
base_window = "60S"

file_path = "loss_function_study.csv"
parameter="loss function"
window_durations = ["MSE", "MAE"]
base_window = "MSE"
"""

metric_columns = ["MAE", "SAE"]
df = pd.read_csv(file_path)

show(df)

# Pivot the data to get one row per (appliance, method), and one column per metric+window
pivot_df = df.pivot_table(
    index=["appliance", "method"],
    columns=parameter,
    values=metric_columns
)

# Flatten column names
pivot_df.columns = [f"{metric}_{window}" for metric, window in pivot_df.columns]
pivot_df.reset_index(inplace=True)

show(pivot_df)

if not dev_flag:
    # === rearrange the values
    for metric in metric_columns:
        for window in window_durations:
            target_col = f"{metric}_{window}"
            deviation_col = f"{metric}_{window}"
            pivot_df[deviation_col] = pivot_df[target_col] 

    # === PIVOT TO WIDE FORMAT ===
    # Keep only relevant columns (appliance, method, and relative deviations)
    columns_to_keep = ["appliance", "method"]
    for metric in metric_columns:
        for window in window_durations:
            columns_to_keep.append(f"{metric}_{window}")
    pivot_df = pivot_df[columns_to_keep]

    # Pivot method to columns, appliance stays as row
    wide_df = pivot_df.pivot(index="appliance", columns="method")

    # Flatten new MultiIndex columns
    wide_df.columns = [f"{metric}_{method}" for metric, method in wide_df.columns]
    wide_df.reset_index(inplace=True)

    # === REORDER COLUMNS ===
    ordered_columns = ["appliance"]
    for method in ["seq2seq", "seq2point"]:  # Order of methods
        for window in window_durations:
            for metric in metric_columns:
                ordered_columns.append(f"{metric}_{window}_{method}")

    # Reorder and round
    wide_df = wide_df[ordered_columns]
    wide_df = wide_df.round(2)

    # === REORDER ROWS BY APPLIANCE ORDER ===
    appliance_order = ["kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"]
    wide_df["appliance"] = pd.Categorical(wide_df["appliance"], categories=appliance_order, ordered=True)
    final_df = wide_df.sort_values("appliance").reset_index(drop=True)

    show(final_df)
    
else:
    # === CALCULATE RELATIVE DEVIATIONS ===
    # Calculate relative deviation from base_window for each metric and target window
    for metric in metric_columns:
        base_col = f"{metric}_{base_window}"
        for window in window_durations:
            if window == base_window:
                continue  # Skip base itself
            target_col = f"{metric}_{window}"
            deviation_col = f"{metric}_{window}"
            pivot_df[deviation_col] = ((pivot_df[target_col] - pivot_df[base_col]) / pivot_df[base_col]) * 100


    # === PIVOT TO WIDE FORMAT ===
    # Keep only relevant columns (appliance, method, and relative deviations)
    columns_to_keep = ["appliance", "method"]
    for metric in metric_columns:
        for window in window_durations:
            if window != base_window:
                columns_to_keep.append(f"{metric}_{window}")
    pivot_df = pivot_df[columns_to_keep]

    # Pivot method to columns, appliance stays as row
    wide_df = pivot_df.pivot(index="appliance", columns="method")

    # Flatten new MultiIndex columns
    wide_df.columns = [f"{metric}_{method}" for metric, method in wide_df.columns]
    wide_df.reset_index(inplace=True)

    # === REORDER COLUMNS ===
    ordered_columns = ["appliance"]
    for method in ["seq2point", "seq2seq"]:  # Order of methods
        for window in window_durations:
            if window == base_window:
                continue
            for metric in metric_columns:
                ordered_columns.append(f"{metric}_{window}_{method}")

    # Reorder and round
    wide_df = wide_df[ordered_columns]
    wide_df = wide_df.round(2)

    # === REORDER ROWS BY APPLIANCE ORDER ===
    appliance_order = ["kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"]
    wide_df["appliance"] = pd.Categorical(wide_df["appliance"], categories=appliance_order, ordered=True)
    deviation_df = wide_df.sort_values("appliance").reset_index(drop=True)

    show(deviation_df)