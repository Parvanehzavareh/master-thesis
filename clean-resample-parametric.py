import pandas as pd
from pandasgui import show
import numpy as np

def get_anomalies(df):
    #filter your DataFrame to get rows where the sum of all appliance power columns (excluding "aggregate") is greater than the "aggregate" column.
    sum_appliances = df.drop(columns=["aggregate"]).sum(axis=1)
    df_anomalies = df[sum_appliances > df["aggregate"]]
    return df_anomalies

def get_anomalies_2(df):
    #DataFrame of rows where any column (except "aggregate") has a value greater than "aggregate"
    mask = (df.drop(columns=["aggregate"]).gt(df["aggregate"], axis=0)).any(axis=1)
    return df[mask]

def clean_resample(file_path, dict, resample_time="60S"):
    df_house = pd.read_csv(file_path)
    
    #% only focusing on the main 5 items
    df_house = df_house[list(dict.keys())].rename(columns=dict)

    #% handling missing data
    # Compute time gaps
    time_diff = df_house['time'].diff()
    df_house_filled = df_house.copy()
    df_house_filled[time_diff > 180] = 0  # Set to zero if gap > 3 min
    df_house_filled.fillna(method="ffill", inplace=True)  # Fill forward otherwise
    #df_house_filled.fillna(method="bfill", inplace=True)
    df_house_filled.fillna(0, inplace=True) # set nan to zero for the beggining

    #% changing time to data&time
    df_house_filled['time'] = pd.to_datetime(df_house['time'], unit="s")
    df_house_filled.set_index('time', inplace=True)

    #% resampling the data
    df_resampled = df_house_filled.resample(resample_time).mean()
    # .mean() takes the mean of the data for each resample_time (e.g. 60S) window.
    
    #Drop rows where all columns are NaN for extended periods
    df_resampled = df_resampled.dropna(how="all")
    return df_resampled

def normalize_1(df):  #std for aggregate + ratio for appliance
    df_normalized = df.copy()
    appliance_columns = df.columns[df.columns != "aggregate"]
    df_normalized[appliance_columns] = df_normalized[appliance_columns].div(df_normalized["aggregate"], axis=0)
    df_normalized["aggregate"] = (df_normalized["aggregate"] - df_normalized["aggregate"].mean()) / df_normalized["aggregate"].std()
    df_normalized = df_normalized.replace([np.nan, np.inf, -np.inf], 0)
    return df_normalized

def normalize_2(df): # min_max for aggregate + ratio for appliance
    df_normalized = df.copy()
    appliance_columns = df.columns[df.columns != "aggregate"]
    df_normalized[appliance_columns] = df_normalized[appliance_columns].div(df_normalized["aggregate"], axis=0)
    df_normalized["aggregate"] = (df_normalized['aggregate'] - df_normalized['aggregate'].min()) / (df_normalized['aggregate'].max() - df_normalized['aggregate'].min())
    df_normalized = df_normalized.replace([np.nan, np.inf, -np.inf], 0)
    return df_normalized

def normalize_3(df): # std for aggregate + std for appliance
    df_normalized = df.copy()
    appliance_columns = df.columns[df.columns != "aggregate"]
    df_normalized[appliance_columns] = (df_normalized[appliance_columns] - df_normalized[appliance_columns].mean()) / df_normalized[appliance_columns].std()
    df_normalized["aggregate"] = (df_normalized["aggregate"] - df_normalized["aggregate"].mean()) / df_normalized["aggregate"].std()
    df_normalized = df_normalized.replace([np.nan, np.inf, -np.inf], 0)
    return df_normalized

def normalize_4(df): # min_max for aggregate + min_max for appliance
    df_normalized = df.copy()
    appliance_columns = df.columns[df.columns != "aggregate"]
    df_normalized[appliance_columns] = (df_normalized[appliance_columns] - df_normalized[appliance_columns].min()) / (df_normalized[appliance_columns].max() - df_normalized[appliance_columns].min())
    df_normalized["aggregate"] = (df_normalized['aggregate'] - df_normalized['aggregate'].min()) / (df_normalized['aggregate'].max() - df_normalized['aggregate'].min())
    df_normalized = df_normalized.replace([np.nan, np.inf, -np.inf], 0)
    return df_normalized

def get_activation_old(df):
    on_power_threshold = {"kettle" : 2000,
                          "microwave" : 200,
                          "fridge_freezer" : 50,
                          "dishwasher" : 10,
                          "washer_dryer" : 20,}
    df_on_off = df.copy()
    df_on_off.drop(columns=["aggregate"], inplace=True)
    
    for appliance, threshold in on_power_threshold.items():
        if appliance in df.columns:
            df_on_off[appliance] = (df_on_off[appliance] > threshold).astype(int)
    return df_on_off


def get_activation(df):
    """
    Converts power readings into ON/OFF states with appliance-specific minimum ON and OFF durations.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing appliance power readings.

    Returns:
        pd.DataFrame: ON/OFF state DataFrame with enforced minimum durations.
    """
    # Power thresholds for ON/OFF classification from articles
    on_power_threshold = {
        "kettle": 2000,   #1000
        "microwave": 200,  #50
        "fridge_freezer": 50,  #5
        "dishwasher": 10,   #10
        "washer_dryer": 20   #20
    }

    # Appliance-specific min ON/OFF durations (in number of samples)
    min_on_duration = {
        "kettle": 12,   #20       # Kettle must be ON for at least 10 samples
        "microwave": 12,   #10     # Microwave must be ON for at least 5 samples
        "fridge_freezer": 60, #60 # Fridge must be ON for at least 20 samples
        "dishwasher": 1800,  #1800    # Dishwasher must be ON for at least 15 samples
        "washer_dryer": 1800  #1800   # Washer/Dryer must be ON for at least 30 samples
    }
    
    min_off_duration = {
        "kettle": 0,   #10        # Kettle must be OFF for at least 5 samples
        "microwave": 30,  #10      # Microwave must be OFF for at least 3 samples
        "fridge_freezer": 12, #60 # Fridge must be OFF for at least 10 samples
        "dishwasher": 1800,   #300   # Dishwasher must be OFF for at least 10 samples
        "washer_dryer": 160   #300  # Washer/Dryer must be OFF for at least 15 samples
    }
    
    
    
    
    
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

    # Copy DataFrame and remove 'aggregate' column
    df_on_off = df.copy()
    df_on_off.drop(columns=["aggregate"], inplace=True)

    for appliance, threshold in on_power_threshold.items():
        if appliance in df.columns:
            # Convert to ON/OFF states
            df_on_off[appliance] = (df_on_off[appliance] > threshold).astype(int)

            # Apply minimum ON/OFF duration filtering
            df_on_off[appliance] = enforce_min_duration(
                df_on_off[appliance], 
                min_on=min_on_duration.get(appliance, 5), 
                min_off=min_off_duration.get(appliance, 5)
            )

    return df_on_off


def enforce_min_duration(series, min_on, min_off):
    """
    Ensures an appliance remains ON for at least `min_on` samples and OFF for at least `min_off` samples.
    
    Parameters:
        series (pd.Series): A binary 1/0 series representing appliance ON/OFF states.
        min_on (int): Minimum ON duration in samples.
        min_off (int): Minimum OFF duration in samples.
    
    Returns:
        pd.Series: Adjusted ON/OFF series.
    """
    series = series.copy()
    prev_state = series.iloc[0]
    count = 0

    for i in range(1, len(series)):
        if series.iloc[i] == prev_state:
            count += 1
        else:
            if prev_state == 1 and count < min_on:
                series.iloc[i - count : i] = 0  # Reset short ON period to OFF
            elif prev_state == 0 and count < min_off:
                series.iloc[i - count : i] = 1  # Reset short OFF period to ON
            count = 1
            prev_state = series.iloc[i]

    return series


normalize_list = normalize_1, normalize_2
normal_name_list = "std_ratio", "min_max_ratio"
resample_time_list = "30S", "45S", "60S"

for i in range(2):
    normalize = normalize_list[i]
    normal_name = normal_name_list[i]
    for j in range(3):
        resample_time = resample_time_list[j]

        #% house 5
        file_path = r"C:\master\data\house_5\merged_appliances.csv"
        dict_5 = {"time" : "time",
                "channel_1_power" : "aggregate",
                "channel_18_power" : "kettle",
                "channel_19_power" : "fridge_freezer", 
                "channel_22_power" : "dishwasher",
                "channel_23_power" : "microwave",
                "channel_24_power" : "washer_dryer",}
        df_5 = clean_resample(file_path, dict_5, resample_time=resample_time)
        print(df_5.isnull().sum().sum())
        df_5_anomalies = get_anomalies(df_5)
        df_5_anomalies_2 = get_anomalies_2(df_5)
        df_5.to_csv(fr"C:\master\data\house_5\resampled_data_{normal_name}_{resample_time}.csv")
        df_5_normalized = normalize(df_5)
        df_5_normalized.to_csv(fr"C:\master\data\house_5\normalized_data_{normal_name}_{resample_time}.csv")
        df_5_activation = get_activation(df_5)
        df_5_activation.to_csv(fr"C:\master\data\house_5\activation_data_{normal_name}_{resample_time}.csv")

        #show(df_5.head(500))
        #show(df_5_normalized.head(500))
        #show(df_5_activation.head(500))
        #show(df_5_anomalies)
        #show(df_5_anomalies_2)


        #% house 4
        file_path = r"C:\master\data\house_4\merged_appliances.csv"
        dict_4 = {"time" : "time",
                "channel_1_power" : "aggregate",
                "channel_3_power" : "kettle",
                "channel_5_power" : "fridge_freezer", }
        df_4 = clean_resample(file_path, dict_4, resample_time=resample_time)
        print(df_4.isnull().sum().sum())
        df_4_anomalies = get_anomalies(df_4)
        df_4_anomalies_2 = get_anomalies_2(df_4)
        df_4.to_csv(fr"C:\master\data\house_4\resampled_data_{normal_name}_{resample_time}.csv")
        df_4_normalized = normalize(df_4)
        df_4_normalized.to_csv(fr"C:\master\data\house_4\normalized_data_{normal_name}_{resample_time}.csv")
        df_4_activation = get_activation(df_4)
        df_4_activation.to_csv(fr"C:\master\data\house_4\activation_data_{normal_name}_{resample_time}.csv")

        #show(df_4.head(500))
        #show(df_4_normalized.head(500))
        #show(df_4_activation.head(500))
        #show(df_4_anomalies)
        #show(df_4_anomalies_2)


        #% house 3
        file_path = r"C:\master\data\house_3\merged_appliances.csv"
        dict_3 = {"time" : "time",
                "channel_1_power" : "aggregate",
                "channel_2_power" : "kettle", }
        df_3 = clean_resample(file_path, dict_3, resample_time=resample_time)
        print(df_3.isnull().sum().sum())
        df_3_anomalies = get_anomalies(df_3)
        df_3_anomalies_2 = get_anomalies_2(df_3)
        df_3.to_csv(fr"C:\master\data\house_3\resampled_data_{normal_name}_{resample_time}.csv")
        df_3_normalized = normalize(df_3)
        df_3_normalized.to_csv(fr"C:\master\data\house_3\normalized_data_{normal_name}_{resample_time}.csv")
        df_3_activation = get_activation(df_3)
        df_3_activation.to_csv(fr"C:\master\data\house_3\activation_data_{normal_name}_{resample_time}.csv")

        #show(df_3.head(500))
        #show(df_3_normalized.head(500))
        #show(df_3_activation.head(500))
        #show(df_3_anomalies)
        #show(df_3_anomalies_2)

        #% house 2
        file_path = r"C:\master\data\house_2\merged_appliances.csv"
        dict_2 = {"time" : "time",
                "channel_1_power" : "aggregate",
                "channel_8_power" : "kettle", 
                "channel_12_power" : "washer_dryer",
                "channel_13_power" : "dishwasher",
                "channel_14_power" : "fridge_freezer",
                "channel_15_power" : "microwave",}
        df_2 = clean_resample(file_path, dict_2, resample_time=resample_time)
        print(df_2.isnull().sum().sum())
        df_2_anomalies = get_anomalies(df_2)
        df_2_anomalies_2 = get_anomalies_2(df_2)
        df_2.to_csv(fr"C:\master\data\house_2\resampled_data_{normal_name}_{resample_time}.csv")
        df_2_normalized = normalize(df_2)
        df_2_normalized.to_csv(fr"C:\master\data\house_2\normalized_data_{normal_name}_{resample_time}.csv")
        df_2_activation = get_activation(df_2)
        df_2_activation.to_csv(fr"C:\master\data\house_2\activation_data_{normal_name}_{resample_time}.csv")

        #show(df_2.head(500))
        #show(df_2_normalized.head(500))
        #show(df_2_activation.head(500))
        #show(df_2_anomalies)
        #show(df_2_anomalies_2)

        #%% house 1
        file_path = r"C:\master\data\house_1\merged_appliances.csv"
        dict_1 = {"time" : "time",
                "channel_1_power" : "aggregate",
                "channel_10_power" : "kettle", 
                "channel_5_power" : "washer_dryer",
                "channel_6_power" : "dishwasher",
                "channel_12_power" : "fridge_freezer",
                "channel_13_power" : "microwave",}
        df_1 = clean_resample(file_path, dict_1, resample_time=resample_time)
        print(df_1.isnull().sum().sum())
        df_1_anomalies = get_anomalies(df_1)
        df_1_anomalies_2 = get_anomalies_2(df_1)
        df_1.to_csv(fr"C:\master\data\house_1\resampled_data_{normal_name}_{resample_time}.csv")
        df_1_normalized = normalize(df_1)
        df_1_normalized.to_csv(fr"C:\master\data\house_1\normalized_data_{normal_name}_{resample_time}.csv")
        df_1_activation = get_activation(df_1)
        df_1_activation.to_csv(fr"C:\master\data\house_1\activation_data_{normal_name}_{resample_time}.csv")

        #show(df_1.head(500))
        #show(df_1_normalized.head(500))
        #show(df_1_activation.head(500))
        #show(df_1_anomalies)
        #show(df_1_anomalies_2)


