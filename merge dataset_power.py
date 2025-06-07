import pandas as pd
from pandasgui import show
from functools import reduce

def merge_appliance_data(folder_path, file_names):
    """
    Reads appliance data files, merges them into one DataFrame, and saves the merged DataFrame to a CSV file.

    Parameters:
        folder_path (str): Path to the folder containing the data files.
        file_names (list): List of file names to process.

    Returns:
        pd.DataFrame: Merged DataFrame containing all appliance data.
    """
    # Dictionary to hold DataFrames
    appliance_data = {}
    column_names = ['time', 'power']

    # Loop through the file names to create the dictionary
    for file_name in file_names:
        # Extract appliance name from the file name (e.g., "channel_1" from "channel_1.dat")
        appliance_name = file_name.split(".")[0]
        
        # Construct full file path
        file_path = f"{folder_path}\\{file_name}"

        # Read the file into a DataFrame
        df = pd.read_csv(file_path, sep=' ', header=None, names=column_names)
        df = df.rename(columns={"power": appliance_name + "_power"})
        
        # Add the DataFrame to the dictionary
        appliance_data[appliance_name] = df

    # Merge all DataFrames on 'time' using reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="time", how="outer"), list(appliance_data.values()))

    return merged_df



#%% house 1
# Provide the path to your CSV file
folder_path = r"C:\master\data\house_1"
file_names = [
    "channel_1.dat", "channel_5.dat", "channel_6.dat",
    "channel_10.dat", "channel_12.dat", "channel_13.dat",
]

merged_df_house1 = merge_appliance_data(folder_path, file_names)

# Display the dictionary keys (appliance names)
show(merged_df_house1.head(500))

output_file = folder_path + "\\merged_appliances.csv"
merged_df_house1.to_csv(output_file, index=False)


#%% house 2
# Provide the path to your CSV file
folder_path = r"C:\master\data\house_2"
file_names = [
    "channel_1.dat", "channel_2.dat", "channel_3.dat",
    "channel_4.dat", "channel_5.dat", "channel_6.dat",
    "channel_7.dat", "channel_8.dat", "channel_9.dat",
    "channel_10.dat", "channel_11.dat", "channel_12.dat",
    "channel_13.dat", "channel_14.dat", "channel_15.dat",
    "channel_16.dat", "channel_17.dat", "channel_18.dat", 
    "channel_19.dat"
]

merged_df_house2 = merge_appliance_data(folder_path, file_names)

# Display the dictionary keys (appliance names)
show(merged_df_house2.head(500))

output_file = folder_path + "\\merged_appliances.csv"
merged_df_house2.to_csv(output_file, index=False)


#%% house 3
# Provide the path to your CSV file
folder_path = r"C:\master\data\house3"
file_names = [
    "channel_1.dat", "channel_2.dat", "channel_3.dat",
    "channel_4.dat", "channel_5.dat"
]

merged_df_house3 = merge_appliance_data(folder_path, file_names)

# Display the dictionary keys (appliance names)
show(merged_df_house3.head(500))

output_file = folder_path + "\\merged_appliances.csv"
merged_df_house3.to_csv(output_file, index=False)


#%% house 4
# Provide the path to your CSV file
folder_path = r"C:\master\data\house_4"
file_names = [
    "channel_1.dat", "channel_2.dat", "channel_3.dat",
    "channel_4.dat", "channel_5.dat", "channel_6.dat"
]

merged_df_house4 = merge_appliance_data(folder_path, file_names)

# Display the dictionary keys (appliance names)
show(merged_df_house4.head(500))

output_file = folder_path + "\\merged_appliances.csv"
merged_df_house4.to_csv(output_file, index=False)


#%% house 5
# Provide the path to your CSV file
folder_path = r"C:\master\data\house_5"
file_names = [
    "channel_1.dat", "channel_2.dat", "channel_3.dat",
    "channel_4.dat", "channel_5.dat", "channel_6.dat",
    "channel_7.dat", "channel_8.dat", "channel_9.dat",
    "channel_10.dat", "channel_11.dat", "channel_12.dat",
    "channel_13.dat", "channel_14.dat", "channel_15.dat",
    "channel_16.dat", "channel_17.dat", "channel_18.dat", 
    "channel_19.dat", "channel_20.dat", "channel_21.dat", 
    "channel_22.dat", "channel_23.dat", "channel_24.dat", 
    "channel_25.dat"
]

merged_df_house5 = merge_appliance_data(folder_path, file_names)

# Display the dictionary keys (appliance names)
show(merged_df_house5.head(500))

output_file = folder_path + "\\merged_appliances.csv"
merged_df_house5.to_csv(output_file, index=False)

