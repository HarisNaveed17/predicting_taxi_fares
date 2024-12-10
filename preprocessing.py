import os
import numpy as np
import pandas as pd
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


DATA_DIR = "/scratch/gl2758/PTSA/Data"
PROC_DATA_DIR = "/scratch/gl2758/PTSA/output"

def get_taxi_zone_id(area_names):
    """Retrieve the Taxi Zone ID used in the database for areas of interest

    Args:
        area_names (list): list of names of areas of interest

    Returns:
        DataFrame: DataFrame of LocationIDs of our areas of interest
    """
    zones = pd.read_csv("taxi_zone_lookup.csv")
    req_zones = zones[zones["Zone"].isin(area_names)]
    for locationCode, locationName in zip(req_zones["LocationID"], req_zones["Zone"]):
        print(f"The {locationName} area has the locationID: {locationCode}")
    return req_zones["LocationID"].values.tolist()


def get_holidays(df):
    df['is_weekend'] = np.where(df.index.weekday >= 5, 1, 0)
    spring_2022 = ['2022-03-14', '2022-03-20']
    spring_2023 = ['2023-03-13', '2023-03-19']
    sp22_dates = pd.date_range(spring_2022[0], spring_2022[-1], freq='D').tolist()
    sp23_dates = pd.date_range(spring_2023[0], spring_2023[-1], freq='D').tolist()
    custom_holidays = holidays.HolidayBase()
    custom_holidays.append(sp22_dates)
    custom_holidays.append(sp23_dates)
    df["date"] = pd.to_datetime(df.index)
    df['is_holiday'] = df["date"].apply(lambda x: 1 if x in custom_holidays else 0)
    df = df.drop(columns=["date"])
    return df



def concat_process_data(path, area_names, resample_interval):
    """load parquets files, combine them and export as csv

    Args:
        path (string): path to data folder

    Returns:
        DataFrame: DataFrame object consisting of combined data
    """
    data = list()
    zones = get_taxi_zone_id(area_names)
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        chunk = process_data(filepath, ["tpep_pickup_datetime", "PULocationID", "tip_amount"], zones, resample_interval).round(2)
    
        data.append(chunk)
    full = pd.concat(data).sort_index()
    data_sem22 = full["2022-01-24 00:00:00": "2022-05-09 23:30:00"]
    data_sem23 = full["2023-01-23 00:00:00": "2023-05-16 23:30:00"]
    data_sems = pd.concat([data_sem22, data_sem23])
    data_sems.to_csv(f"MSG_full_data_{resample_interval}.csv")
    return data_sems



def process_data(file, cols_to_use, locs, resample_interval):
    data = pd.read_parquet(file, columns=cols_to_use, engine="pyarrow")
    if len(locs) < 2:
        data_locs = data[data["PULocationID"] == locs[0]].drop(columns=["PULocationID"], axis=1)
    else:
        data_locs = data[data["PULocationID"].isin(locs)].drop(columns=["PULocationID"], axis=1)
    data_locs["tpep_pickup_datetime"] = pd.to_datetime(data_locs["tpep_pickup_datetime"])
    data_locs = data_locs.set_index("tpep_pickup_datetime", drop=True).sort_index()
    data_hourly = data_locs.resample(resample_interval).size()
    data_fin = data_locs.resample(resample_interval).agg({
        "tip_amount": "median"
    })
    data_fin["Number_of_fares"] = data_hourly
    #data_fin = get_holidays(data_fin)
    return data_fin



if __name__ == "__main__":
    #df = concat_process_data(DATA_DIR)
# <<<<<<< Updated upstream
    area_names = ["Penn Station/Madison Sq West"]
    full_data = concat_process_data(DATA_DIR, area_names, "30min")
    print(full_data)
    # locs = get_taxi_zone_id(area_names)
    # df = process_data("raw_data/yellow_tripdata_2022-01.parquet", ["tpep_pickup_datetime", "PULocationID", "tip_amount"], locs)
    # result = seasonal_decompose(df["Number_of_fares"], model='additive')
    # result.plot()
    #plt.show()

