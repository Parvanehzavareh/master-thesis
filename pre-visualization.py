import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from pandasgui import show
import matplotlib.dates as mdates

# Define houses and file names
root_folder = r"C:\master\data"
houses = ["house_1", "house_2", "house_3", "house_4", "house_5"]
data_types = ["resampled_data", "normalized_data", "activation_data"]
columns = ["aggregate", "kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"]

dataframes = {}

# Load data
for house in houses:
    for data_type in data_types:
        file_name = root_folder + f"\{house}\{data_type}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, index_col=0, parse_dates=True)
            dataframes[f"{house}_{data_type}"] = df

for house in houses:
    df_name = house + "_" + "resampled_data"
    df = dataframes[df_name].copy()
    appliance_sum = df.drop(columns=['aggregate']).sum(axis=1)
    df['other_appliances'] = df['aggregate'] - appliance_sum
    df_name_new = df_name + "_other"
    dataframes[df_name_new] = df

# Visualization Functions
def plot_time_series(df, title, start_time=None, duration=None):
    """
    Plots time series data within a specified time range.

    Parameters:
        df (pd.DataFrame): The dataframe containing time series data.
        title (str): Title of the plot.
        start_time (str or pd.Timestamp, optional): The starting time of the plot.
        duration (str, optional): The duration for which to plot data (e.g., '1H', '2D').
    """
    df = df.drop(columns=["aggregate"], errors='ignore')  # Exclude aggregate column

    # Filter data based on time range if provided
    if start_time is not None and duration is not None:
        end_time = pd.to_datetime(start_time) + pd.to_timedelta(duration)
        df = df.loc[start_time:end_time]

    plt.figure(figsize=(12, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power Consumption (W)")
    plt.show()

def plot_stacked_area(df, title, start_time=None, duration=None):
    """
    Plots a stacked area chart of time series data within a specified time range.

    Parameters:
        df (pd.DataFrame): The dataframe containing time series data.
        title (str): Title of the plot.
        start_time (str or pd.Timestamp, optional): The starting time of the plot.
        duration (str, optional): The duration for which to plot data (e.g., '1H', '2D').
    """
    df = df.drop(columns=["aggregate"], errors='ignore')  # Exclude aggregate column

    # Filter data based on time range if provided
    if start_time is not None and duration is not None:
        end_time = pd.to_datetime(start_time) + pd.to_timedelta(duration)
        df = df.loc[start_time:end_time]

    df.plot(kind='area', stacked=True, figsize=(12, 6), alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power Consumption (W)")
    plt.show()

def plot_histogram(df, title):
    df.hist(figsize=(12, 6), bins=50, alpha=0.7)
    plt.suptitle(title)
    plt.show()

def plot_heatmap(df, title):
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

def plot_boxplot(df, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
    
def plot_time_series_with_range_single_plot(df, start_time, duration, title):
    """
    Plots time series for all columns in the dataframe within the specified time range.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing the time series data.
        start_time (str or pd.Timestamp): Starting timestamp.
        duration (str): Duration in pandas-compatible format (e.g., '1H', '30min', '1D').
        title (str): Title of the plot.
    """
    #df = df.drop(columns=["aggregate"], errors='ignore')  # Exclude aggregate column
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.to_timedelta(duration)
    df_filtered = df.loc[start_time:end_time]
    
    if df_filtered.empty:
        print("No data available for the specified time range.")
        return
    
    plt.figure(figsize=(12, 6))
    for col in df_filtered.columns:
        plt.plot(df_filtered.index, df_filtered[col], label=col)
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Power Consumption (W)")
    plt.show()

    
def plot_time_series_subplots(df, start_time, duration, title):
    """
    Plots each time series in a separate subplot within a specified time range.

    Parameters:
        df (pd.DataFrame): The dataframe containing time series data.
        start_time (str or pd.Timestamp): The starting time of the plot.
        duration (str): The duration for which to plot data (e.g., '1H', '2D').
        title (str): Title of the overall figure.
    """
    # Select the time range
    end_time = pd.to_datetime(start_time) + pd.to_timedelta(duration)
    df_filtered = df.loc[start_time:end_time]

    # Drop aggregate if it exists
    #df_filtered = df_filtered.drop(columns=["aggregate"], errors="ignore")

    # Create subplots
    num_plots = len(df_filtered.columns)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot case

    for ax, col in zip(axes, df_filtered.columns):
        ax.plot(df_filtered.index, df_filtered[col], label=col)
        ax.set_title(f"{title} - {col}")
        ax.set_ylabel("Power Consumption (W)")
        ax.legend()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def plot_dual_axis(power_df, activation_df, appliance, start_time, duration):
    """
    Creates a dual-axis plot for power and activation time series.
    
    Parameters:
      power_df (pd.DataFrame): DataFrame containing power data with DateTimeIndex.
      activation_df (pd.DataFrame): DataFrame containing activation data with DateTimeIndex.
      appliance (str): Column name for the appliance to plot.
      start_time (str or pd.Timestamp): The starting time for the plot.
      duration (str): Duration for the plot (e.g., "1H" for one hour, "30min" for 30 minutes).
    """
    # Convert start_time to datetime if it's not already
    start_time = pd.to_datetime(start_time)
    # Compute end time using Pandas Timedelta
    end_time = start_time + pd.Timedelta(duration)
    
    # Filter the DataFrames for the specified time interval
    power_segment = power_df.loc[start_time:end_time]
    activation_segment = activation_df.loc[start_time:end_time]
    
    # Create the dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()
    
    ax1.plot(power_segment.index, power_segment[appliance], color='blue', label='Power')
    ax2.plot(activation_segment.index, activation_segment[appliance], color='red', label='Activation')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Power', color='blue')
    ax2.set_ylabel('Activation', color='red')
    plt.title(f"Dual-Axis Plot for {appliance}\nfrom {start_time} for duration {duration}")
    
    # Improve layout and show plot
    fig.tight_layout()
    plt.show()


def plot_shared_subplots(power_df, activation_df, appliance, start_time, duration):
    """
    Creates a plot with two subplots (power and activation) sharing the same time axis.
    
    Parameters:
      power_df (pd.DataFrame): DataFrame containing power data with DateTimeIndex.
      activation_df (pd.DataFrame): DataFrame containing activation data with DateTimeIndex.
      appliance (str): Column name for the appliance to plot.
      start_time (str or pd.Timestamp): The starting time for the plot.
      duration (str): Duration for the plot (e.g., "1H" for one hour, "30min" for 30 minutes).
    """
    start_time = pd.to_datetime(start_time)
    end_time = start_time + pd.Timedelta(duration)
    
    power_segment = power_df.loc[start_time:end_time]
    activation_segment = activation_df.loc[start_time:end_time]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8))
    
    ax1.plot(power_segment.index, power_segment[appliance], color='blue', label='Power')
    ax1.set_ylabel('Power')
    ax1.legend()
    
    ax2.plot(activation_segment.index, activation_segment[appliance], color='red', label='Activation')
    ax2.set_ylabel('Activation')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    plt.suptitle(f"Shared Subplots for {appliance}\nfrom {start_time} for duration {duration}", y=1.02)
    plt.tight_layout()
    plt.show()




def plot_overlaid_timeseries(dfs, start_times, appliance, duration, house_labels=None):
    """
    Plots an overlaid time series for a selected appliance from multiple houses,
    shifting each house's time series so that its given start time becomes 0.
    
    Parameters:
      dfs (list of pd.DataFrame): List of DataFrames with DateTimeIndex containing power data.
      start_times (list): List of starting times (strings or pd.Timestamp) for each DataFrame.
      appliance (str): Column name for the appliance power data.
      duration (str): Duration for the plot (e.g., "1H", "30min").
      house_labels (list): Optional list of labels for the houses. If None, defaults to House 1, House 2, ...
    
    The x-axis will represent relative time in seconds from each house's starting point.
    """
    if house_labels is None:
        house_labels = [f"House {i+1}" for i in range(len(dfs))]
    
    # Convert duration to a Timedelta and get total seconds.
    dur_td = pd.Timedelta(duration)
    dur_seconds = dur_td.total_seconds()
    
    plt.figure(figsize=(12, 6))
    for df, start, label in zip(dfs, start_times, house_labels):
        if appliance not in df.columns:
            continue
        # Define the pivot for this DataFrame.
        pivot = pd.to_datetime(start)
        end = pivot + dur_td
        # Extract the segment for the given period.
        segment = df.loc[pivot:end]
        if segment.empty:
            continue
        # Compute relative time (in seconds) from the pivot.
        rel_time = (segment.index - pivot) / pd.Timedelta("1s")
        plt.plot(rel_time, segment[appliance], label=label)
    
    plt.xlabel("Relative Time (seconds)")
    plt.ylabel("Power")
    plt.title(f"Overlaid Time Series for '{appliance}'\n(Each series shifted to start at 0)")
    plt.xlim(0, dur_seconds)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_boxplot_many(dfs, start_times, appliance, duration, house_labels=None):
    """
    Creates a grouped boxplot comparing the distribution of power for a selected appliance 
    across houses for the same relative period.
    
    Each DataFrame is first trimmed to the period starting from its given start time up to that time + duration.
    
    Parameters:
      dfs (list of pd.DataFrame): List of DataFrames with DateTimeIndex.
      start_times (list): List of starting times (strings or pd.Timestamp) for each DataFrame.
      appliance (str): Column name for the appliance power data.
      duration (str): Duration for the period (e.g., "1H", "30min").
      house_labels (list): Optional list of labels for the houses.
    """
    if house_labels is None:
        house_labels = [f"House {i+1}" for i in range(len(dfs))]
    
    data_to_plot = []
    valid_labels = []
    dur_td = pd.Timedelta(duration)
    
    for df, start, label in zip(dfs, start_times, house_labels):
        if appliance not in df.columns:
            continue
        pivot = pd.to_datetime(start)
        end = pivot + dur_td
        segment = df.loc[pivot:end]
        values = segment[appliance].dropna().values
        if len(values) == 0:
            continue
        data_to_plot.append(values)
        valid_labels.append(label)
    
    if not data_to_plot:
        print("No valid data available for boxplot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=valid_labels)
    plt.xlabel("House")
    plt.ylabel("Power")
    plt.title(f"Boxplot Comparison for '{appliance}'\n(Period: relative time 0 to {dur_td})")
    plt.tight_layout()
    plt.show()


def compute_summary_statistics(dfs, start_times, appliance, duration, house_labels=None):
    """
    Computes summary statistics for the selected appliance's power usage across houses,
    using data extracted from each house's starting time up to starting time + duration.
    
    Parameters:
      dfs (list of pd.DataFrame): List of DataFrames with DateTimeIndex.
      start_times (list): List of starting times (strings or pd.Timestamp) for each DataFrame.
      appliance (str): Column name for the appliance power data.
      duration (str): Duration for the period (e.g., "1H", "30min").
      house_labels (list): Optional list of labels for the houses.
    
    Returns:
      pd.DataFrame: A table with summary statistics (mean, median, std, min, max, count) for each house.
    """
    if house_labels is None:
        house_labels = [f"House {i+1}" for i in range(len(dfs))]
    
    stats_list = []
    dur_td = pd.Timedelta(duration)
    
    for label, df, start in zip(house_labels, dfs, start_times):
        if appliance not in df.columns:
            continue
        pivot = pd.to_datetime(start)
        end = pivot + dur_td
        segment = df.loc[pivot:end][appliance].dropna()
        if segment.empty:
            continue
        stats = {
            "House": label,
            "Mean": segment.mean(),
            "Median": segment.median(),
            "Std": segment.std(),
            "Min": segment.min(),
            "Max": segment.max(),
            "Count": segment.count()
        }
        stats_list.append(stats)
    
    if not stats_list:
        print("No valid data available for summary statistics.")
        return pd.DataFrame()
    
    stats_df = pd.DataFrame(stats_list)
    return stats_df


def plot_facet_timeseries(dfs, start_times, appliance, duration, house_labels=None):
    """
    Creates a multi-panel (facet) plot of the time series for the selected appliance from each house,
    shifting each series so that its given starting time becomes 0.
    
    Each subplot will display relative time (in seconds) on the x-axis, and all subplots share the same x-axis range.
    
    Parameters:
      dfs (list of pd.DataFrame): List of DataFrames with DateTimeIndex.
      start_times (list): List of starting times (strings or pd.Timestamp) for each DataFrame.
      appliance (str): Column name for the appliance power data.
      duration (str): Duration for the period (e.g., "1H", "30min").
      house_labels (list): Optional list of labels for the houses.
    """
    if house_labels is None:
        house_labels = [f"House {i+1}" for i in range(len(dfs))]
    
    dur_td = pd.Timedelta(duration)
    dur_seconds = dur_td.total_seconds()
    
    valid_segments = []
    valid_labels = []
    
    # Extract and shift each segment.
    for df, start, label in zip(dfs, start_times, house_labels):
        if appliance not in df.columns:
            continue
        pivot = pd.to_datetime(start)
        end = pivot + dur_td
        segment = df.loc[pivot:end]
        if segment.empty:
            continue
        # Compute relative time in seconds.
        rel_time = (segment.index - pivot) / pd.Timedelta("1s")
        valid_segments.append((rel_time, segment[appliance]))
        valid_labels.append(label)
    
    if not valid_segments:
        print("No valid data available for faceted plot.")
        return
    
    n = len(valid_segments)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4*n), sharex=True)
    if n == 1:
        axes = [axes]
    
    for ax, (rel_time, values), label in zip(axes, valid_segments, valid_labels):
        ax.plot(rel_time, values, label=label)
        ax.set_ylabel("Power")
        ax.set_title(label)
        ax.legend()
        ax.set_xlim(0, dur_seconds)
    
    axes[-1].set_xlabel("Relative Time (second)")
    plt.suptitle(f"Faceted Time Series for '{appliance}'\n(Each series shifted so that its start is 0)", y=0.92)
    plt.tight_layout()
    plt.show()




def plot_data_availability(dict_of_dfs, keys_list=None, fixed_gap_threshold=pd.Timedelta("60s")):
    """
    Plots a timeline for each DataFrame (house) showing available data periods.
    The x-axis spans from the earliest date to the latest date across all DataFrames.
    A fixed gap threshold is used: if the gap between consecutive timestamps
    exceeds this value, it is considered a break.
    
    Parameters:
      dict_of_dfs (dict): Dictionary mapping keys (e.g., house names) to pandas DataFrames,
                          each having a DateTimeIndex.
      keys_list (list, optional): List of keys to include. If None, all keys are used.
      fixed_gap_threshold (pd.Timedelta, optional): The maximum allowed gap between consecutive timestamps 
                          (e.g., 60 seconds). Gaps larger than this value are considered breaks.
    """
    # Use all keys if none are provided
    if keys_list is None:
        keys_list = list(dict_of_dfs.keys())
    
    # Determine global minimum and maximum timestamps across all DataFrames
    global_start = None
    global_end = None
    for key in keys_list:
        if key not in dict_of_dfs:
            continue
        df = dict_of_dfs[key]
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            continue
        if df.empty:
            continue
        df_start = df.index.min()
        df_end = df.index.max()
        if (global_start is None) or (df_start < global_start):
            global_start = df_start
        if (global_end is None) or (df_end > global_end):
            global_end = df_end
    
    if global_start is None or global_end is None:
        print("No valid datetime data found.")
        return

    # Set up the plot
    n = len(keys_list)
    fig, ax = plt.subplots(figsize=(12, n * 0.8 + 2))
    
    y = 0.5  # starting y-position
    for key in keys_list:
        if key not in dict_of_dfs:
            print(f"Warning: Key '{key}' not found in dictionary. Skipping.")
            continue
        
        df = dict_of_dfs[key]
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            print(f"Warning: DataFrame for key '{key}' does not have a DateTimeIndex. Skipping.")
            continue
        
        # Ensure the index is sorted
        timestamps = df.index.sort_values()
        if timestamps.empty:
            print(f"Warning: DataFrame for '{key}' is empty. Skipping.")
            continue
        
        # Identify continuous segments using the fixed gap threshold.
        segments = []
        start_seg = timestamps[0]
        prev_time = timestamps[0]
        for t in timestamps[1:]:
            # If the gap is greater than the fixed threshold, end current segment.
            if (t - prev_time) > fixed_gap_threshold:
                segments.append((start_seg, prev_time))
                start_seg = t
            prev_time = t
        segments.append((start_seg, timestamps[-1]))
        
        # Plot each segment as a horizontal bar.
        for seg in segments:
            start_dt, end_dt = seg
            start_num = mdates.date2num(start_dt)
            width = mdates.date2num(end_dt) - start_num
            ax.broken_barh([(start_num, width)], (y - 0.4, 0.8), facecolors='tab:blue')
        
        # Label the house on the left side of the plot.
        ax.text(mdates.date2num(global_start) - 0.01 * (mdates.date2num(global_end)-mdates.date2num(global_start)),
                y, key, va='center', ha='right', fontsize=10)
        y += 1
    
    # Set x-axis limits based on the global start and end dates.
    ax.set_xlim(mdates.date2num(global_start), mdates.date2num(global_end))
    
    # Format the x-axis as dates.
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    
    ax.set_ylim(0, y)
    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title("Data Availability Periods")
    plt.tight_layout()
    plt.show()



houses = ["house_5", ]
# Generate Plots for Each House
for house in houses:
    normalized_df = dataframes.get(f"{house}_normalized_data")
    activation_df = dataframes.get(f"{house}_activation_data")
    resampled_df = dataframes.get(f"{house}_resampled_data")
    resampled_df_other = dataframes.get(f"{house}_resampled_data_other")

    plot_time_series(resampled_df, f"Time Series - {house} resampled Data",start_time=None, duration=None)
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time=None, duration=None)   
    plot_time_series(resampled_df, f"Time Series - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="6h")
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="6h")
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="24h")
    plot_histogram(normalized_df, f"Histogram - {house} Normalized Data")
    #plot_histogram(resampled_df, f"Histogram - {house} Normalized Data")
    plot_heatmap(normalized_df, f"Correlation Heatmap - {house} Normalized Data")
    plot_boxplot(normalized_df, f"Boxplot - {house} Normalized Data")
    plot_time_series_with_range_single_plot(resampled_df_other, "2014-08-01 00:00:00", "6h", f"Time Series (6H) - {house} Resampled Data")
    #plot_time_series_with_range_single_plot(normalized_df, "2014-08-01 00:00:00", "6h", f"Time Series (6H) - {house} Normalized Data")
    plot_time_series_subplots(resampled_df_other, "2014-09-01 00:00:00", "6h", f"Time Series (1H) - {house} Resampled Data")
    plot_time_series_subplots(resampled_df_other, "2014-09-01 00:00:00", "24h", f"Time Series (1H) - {house} Resampled Data")
    #plot_time_series_subplots(normalized_df, "2014-09-01 00:00:00", "6h", f"Time Series (1H) - {house} Normalized Data")
    
    plot_dual_axis(resampled_df, activation_df, "kettle", "2014-08-01 00:00:00", "24h")
    plot_dual_axis(resampled_df, activation_df, "fridge_freezer", "2014-08-01 00:00:00", "24h")
    plot_dual_axis(resampled_df, activation_df, "dishwasher", "2014-08-01 00:00:00", "24h")
    plot_dual_axis(resampled_df, activation_df, "microwave", "2014-08-01 00:00:00", "24h")
    plot_dual_axis(resampled_df, activation_df, "washer_dryer", "2014-08-01 00:00:00", "24h")
    #plot_shared_subplots(resampled_df, activation_df, "fridge_freezer", "2014-08-01 00:00:00", "24h")


houses = ["house_2", ]
# Generate Plots for Each House
for house in houses:
    normalized_df = dataframes.get(f"{house}_normalized_data")
    activation_df = dataframes.get(f"{house}_activation_data")
    resampled_df = dataframes.get(f"{house}_resampled_data")
    resampled_df_other = dataframes.get(f"{house}_resampled_data_other")

    plot_time_series(resampled_df, f"Time Series - {house} resampled Data",start_time=None, duration=None)
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time=None, duration=None)   
    plot_time_series(resampled_df, f"Time Series - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="6h")
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="6h")
    plot_stacked_area(resampled_df, f"Stacked Area - {house} resampled Data",start_time="2014-09-01 00:00:00", duration="24h")
    plot_histogram(normalized_df, f"Histogram - {house} Normalized Data")
    #plot_histogram(resampled_df, f"Histogram - {house} Normalized Data")
    plot_heatmap(normalized_df, f"Correlation Heatmap - {house} Normalized Data")
    plot_boxplot(normalized_df, f"Boxplot - {house} Normalized Data")
    plot_time_series_with_range_single_plot(resampled_df_other, "2014-08-01 00:00:00", "6h", f"Time Series (6H) - {house} Resampled Data")
    #plot_time_series_with_range_single_plot(normalized_df, "2014-08-01 00:00:00", "6h", f"Time Series (6H) - {house} Normalized Data")
    plot_time_series_subplots(resampled_df_other, "2014-09-01 00:00:00", "6h", f"Time Series (1H) - {house} Resampled Data")
    plot_time_series_subplots(resampled_df_other, "2014-09-01 00:00:00", "24h", f"Time Series (1H) - {house} Resampled Data")
    #plot_time_series_subplots(normalized_df, "2014-09-01 00:00:00", "6h", f"Time Series (1H) - {house} Normalized Data")
    
    plot_dual_axis(resampled_df, activation_df, "kettle", "2013-07-01 6:00:00", "12h")
    plot_dual_axis(resampled_df, activation_df, "fridge_freezer", "2013-07-01 21:00:00", "3h")
    plot_dual_axis(resampled_df, activation_df, "dishwasher", "2013-07-01 21:00:00", "3h")
    plot_dual_axis(resampled_df, activation_df, "microwave", "2013-07-02 5:00:00", "10h")
    plot_dual_axis(resampled_df, activation_df, "washer_dryer", "2013-07-05 21:00:00", "24h")
    #plot_shared_subplots(resampled_df, activation_df, "fridge_freezer", "2014-08-01 00:00:00", "24h")
    

#%% Gnerate plots for all houses simultaneously
key_list = ['house_1_resampled_data', 'house_2_resampled_data', 'house_3_resampled_data', 'house_4_resampled_data', 'house_5_resampled_data']
start_times = ["2013-04-01 00:00:00", "2013-07-01 00:00:00", "2013-03-25 00:00:00", "2013-05-01 00:00:00", "2014-09-01 00:00:00"]
duration = "24h"
appliance = "fridge_freezer"   #["aggregate", "kettle", "fridge_freezer", "dishwasher", "microwave", "washer_dryer"]
dfs = [dataframes[key] for key in key_list if key in dataframes]
dfs_dict = {key : dataframes[key] for key in key_list if key in dataframes}

plot_data_availability(dfs_dict, keys_list=None, fixed_gap_threshold=pd.Timedelta("60s"))

plot_overlaid_timeseries(dfs, start_times, appliance, duration, house_labels=None)
plot_boxplot_many(dfs, start_times, appliance, duration, house_labels=None)
stats_df = compute_summary_statistics(dfs, start_times, appliance, duration, house_labels=None)
plot_facet_timeseries(dfs, start_times, appliance, duration, house_labels=None)

show(stats_df)







#show(dataframes.get("house_5_normalized_data").head(5000))

#show(dataframes.get("house_5_resampled_data").head(5000))

#show(dataframes.get("house_5_activation_data").head(5000))



# Time periods when meters were recording.
# spyder plot for the comparison of different houses and results also



#Time-domain representations of the appliances.
#The time domain characteristics of the current load profiles were examined and presented in Fig. 2. As can be
#observed, there are some similarities between the waveforms of certain appliances, which can reduce the overall
#classification performance of the deep learning model.


# maybe we can do some denoising?
# Histograms of LPH representations of the power signals
#Spectrograms of the appliances
#Short Term Fourier Transform spectrogram for a 3 concurrent load
#sample.




# principal component analysis
