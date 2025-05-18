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
    df_house["time"] = pd.to_datetime(df_house["time"])  # Step 1: Convert
    df_house.set_index("time", inplace=True)       # Step 2: Set as index
    
    #% handling missing data
    # Compute time gaps
    time_diff =df_house.index.to_series().diff()
    gap_mask = time_diff.dt.total_seconds() > 180
   
    
    df_house_filled = df_house.copy()
    df_house_filled[gap_mask] = 0
    
    df_house_filled.fillna(method="ffill", inplace=True)  # Fill forward otherwise
    #df_house_filled.fillna(method="bfill", inplace=True)
    df_house_filled.fillna(0, inplace=True) # set nan to zero for the beggining
    
    
    #% changing time to data&time
    df_house_filled.index = pd.to_datetime(df_house_filled.index, unit="s")

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

def do_all(ind, dict_, normalize, resample_time):
    file_path = rf"C:\Moein\TRANSFER\data\refit\House_{ind}.csv"
    df = clean_resample(file_path, dict_, resample_time=resample_time)
    print(ind, df.isnull().sum().sum())
    df_anomalies = get_anomalies(df)
    df_anomalies_2 = get_anomalies_2(df)
    if normalize.__name__ == "normalize_1":
        normal_name = "std_ratio"
    elif normalize.__name__ == "normalize_2":
        normal_name = "min_max_ratio"
    df.to_csv(rf"C:\Moein\TRANSFER\data\refit\resampled_data_{normal_name}_{resample_time}_house_{ind}.csv")
    df_normalized = normalize(df)
    df_normalized.to_csv(rf"C:\Moein\TRANSFER\data\refit\normalized_data_{normal_name}_{resample_time}_house_{ind}.csv")
    #df_5_activation = get_activation(df_5)
    #df_5_activation.to_csv(r"C:\Moein\TRANSFER\data\house_5\activation_data.csv")
    return df, df_normalized, df_anomalies, df_anomalies_2

normalize_list = [normalize_1, normalize_2]
resample_time_list = ["30S", "45S", "60S"]

for normalize in normalize_list:
    for resample_time in resample_time_list:

        #% House_1
        ind = 1
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance1" : "fridge_freezer", 
                "Appliance6" : "dishwasher",
                "Appliance5" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)
        #show(df.head(500))
        #show(df_normalized.head(500))
        #show(df_anomalies)
        #show(df_anomalies_2)

        #% House_2
        ind = 2
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance8" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance3" : "dishwasher",
                "Appliance5" : "microwave",
                "Appliance2" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)


        #% House_3
        ind = 3
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance2" : "fridge_freezer", 
                "Appliance5" : "dishwasher",
                "Appliance8" : "microwave",
                "Appliance6" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)


        #% House_4
        ind = 4
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance3" : "fridge_freezer", 
                "Appliance8" : "microwave",
                "Appliance4" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)


        #% House_5
        ind = 5
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance8" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance4" : "dishwasher",
                "Appliance7" : "microwave",
                "Appliance3" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_6
        ind = 6
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance7" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance3" : "dishwasher",
                "Appliance6" : "microwave",
                "Appliance2" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_7
        ind = 7
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance6" : "dishwasher",
                "Appliance5" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_8
        ind = 8
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance8" : "microwave",
                "Appliance4" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_9
        ind = 9
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance7" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance4" : "dishwasher",
                "Appliance6" : "microwave",
                "Appliance2" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_10
        ind = 10
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance4" : "fridge_freezer", 
                "Appliance6" : "dishwasher",
                "Appliance8" : "microwave",
                "Appliance5" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_11
        ind = 11
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance7" : "kettle",
                "Appliance2" : "fridge_freezer", 
                "Appliance4" : "dishwasher",
                "Appliance6" : "microwave",
                "Appliance3" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_12
        ind = 12
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance4" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance3" : "microwave",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_13
        ind = 13
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance4" : "dishwasher",
                "Appliance8" : "microwave",
                "Appliance3" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_15
        ind = 15
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance8" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance4" : "dishwasher",
                "Appliance7" : "microwave",
                "Appliance3" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_16
        ind = 16
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance1" : "fridge_freezer", 
                "Appliance6" : "dishwasher",
                "Appliance5" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_17
        ind = 17
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance8" : "kettle",
                "Appliance2" : "fridge_freezer", 
                "Appliance7" : "microwave",
                "Appliance4" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_18
        ind = 18
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance3" : "fridge_freezer", 
                "Appliance6" : "dishwasher",
                "Appliance9" : "microwave",
                "Appliance4" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_19
        ind = 19
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance5" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance4" : "microwave",
                "Appliance2" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_20
        ind = 20
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance9" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance5" : "dishwasher",
                "Appliance8" : "microwave",
                "Appliance4" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)

        #% House_21
        ind = 21
        dict_ = {"Time" : "time",
                "Aggregate" : "aggregate",
                "Appliance7" : "kettle",
                "Appliance1" : "fridge_freezer", 
                "Appliance4" : "dishwasher",
                "Appliance3" : "washer_dryer",}
        df, df_normalized, df_anomalies, df_anomalies_2 = do_all(ind, dict_, normalize, resample_time)
