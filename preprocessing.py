import os
import pandas as pd
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt



DATA_DIR = "raw_data"
PROC_DATA_DIR = "processed_data"

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
    df['is_weekend'] = df.index.weekday >= 5
    spring_2022 = ['2022-03-14', '2022-03-20']
    spring_2023 = ['2023-03-13', '2023-03-19']
    sp22_dates = pd.date_range(spring_2022[0], spring_2022[-1], freq='D').tolist()
    sp23_dates = pd.date_range(spring_2023[0], spring_2023[-1], freq='D').tolist()
    custom_holidays = holidays.HolidayBase()
    custom_holidays.append(sp22_dates)
    custom_holidays.append(sp23_dates)
    df["date"] = pd.to_datetime(df.index)
    df['is_holiday'] = df["date"].apply(lambda x: True if x in custom_holidays else False)
    df = df.drop(columns=["date"])
    return df



def concat_process_data(path, area_names):
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
        chunk = process_data(filepath, ["tpep_pickup_datetime", "PULocationID", "tip_amount"], zones)
    
        data.append(chunk)
    full_data = pd.concat(data)
    full_data.to_csv("full_data.csv")
    return pd.concat(data)



def process_data(file, cols_to_use, locs):
    data = pd.read_parquet(file, columns=cols_to_use, engine="pyarrow")
    if len(locs) < 2:
        data_locs = data[data["PULocationID"] == locs[0]].drop(columns=["PULocationID"], axis=1)
    else:
        data_locs = data[data["PULocationID"].isin(locs)].drop(columns=["PULocationID"], axis=1)
    data_locs["tpep_pickup_datetime"] = pd.to_datetime(data_locs["tpep_pickup_datetime"])
    data_locs = data_locs.set_index("tpep_pickup_datetime", drop=True).sort_index()
    data_hourly = data_locs.resample("30min").size()
    data_fin = data_locs.resample("30min").agg({
        "tip_amount": "sum"
    })
    data_fin["Number_of_fares"] = data_hourly
    data_fin = get_holidays(data_fin)
    return data_fin



if __name__ == "__main__":
    #df = concat_process_data(DATA_DIR)
    area_names = ["Greenwich Village South"]
    full_data = concat_process_data(DATA_DIR, area_names)
    # locs = get_taxi_zone_id(area_names)
    # df = process_data("raw_data/yellow_tripdata_2022-01.parquet", ["tpep_pickup_datetime", "PULocationID", "tip_amount"], locs)
    # result = seasonal_decompose(df["Number_of_fares"], model='additive')
    # result.plot()
    #plt.show()

