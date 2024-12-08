import os
import pandas as pd
import holidays
from datetime import datetime


## Specify neighborhood
def pick_zones(zone=None, borough="Manhattan"):
   
    # Read in the taxi zones
    try:
        zones = pd.read_csv("taxi_zone_lookup.csv")
    except:
        print("Upload the taxi zones .csv in the directory of this script")

    if borough == "Manhattan":
        # Filter data
        zones = zones[zones['Borough'] == 'Manhattan']
        print("See Manhattan borough options...")
        print(zones['Zone'].unique())

        # Save Manhattan location ids
        manhattan_ids = zones['LocationID'].unique()
    else:
        print("WARNING: outside Manhattan. See borough options...")
        print(zones['Zone'].unique())

    if zone:
        # Extract all zones with that zone you pass (could be some hybrid ones)
        neighborhood = zones['Zone'].str.contains(zone)
        results = zones[neighborhood]
        print("See filtered data...")
        print(results)

        # Save location id
        location_ids = results['LocationID'].values

        print("See location ids...")
        print(location_ids)
    else:
        print("WARNING failed zone filter")


    return location_ids, manhattan_ids

## Now specify columns you want to keep
def process_data(file=None, cols_to_keep=None, location_ids=None, manhattan=False, manhattan_ids=None):
    
    # Read in the data
    print("Chunk: ", file)
    try:
        data = pd.read_parquet(file)
    except:
        print("Reload file, error on file: ", file)

    print("Column options...")
    print(data.keys())

    # Filter the data
    if cols_to_keep:
        data = data[cols_to_keep]
    else:
        print("WARNING, keeping all data columns. Watch data size")

    # Isolate to only semester times
    start_dates = ['2022-01-24', '2023-01-23', '2024-01-22']
    end_dates = ['2022-05-17', '2023-05-16', '2024-05-14']

    start_dates = pd.to_datetime(start_dates)
    end_dates = pd.to_datetime(end_dates)

    mask = pd.Series(False, index=data.index)  # Start with all False

    # Apply mask
    for start, end in zip(start_dates, end_dates):
        mask |= data['tpep_pickup_datetime'].between(start, end)

    # Filter
    data = data[mask]

    # Now keep pickup ID in that location
    if len(location_ids) < 2:
        data = data[data["PULocationID"] == location_ids[0]].drop(columns=["PULocationID"], axis=1)
    else:
        data = data[data["PULocationID"].isin(location_ids)].drop(columns=["PULocationID"], axis=1)

    # If you want to keep it Manhattan dropoff only...
    if manhattan:
        data = data[data['DOLocationID'].isin(manhattan_ids)]
    else:
        print("WARNING: not filtering Mahattan dropoffs...")

    # We don't care about dropoff time
    data = data.drop('tpep_dropoff_datetime', axis=1)

    return data

## Now specify parameters for time series
def aggregate_time_series(data=None, interval='30min', agg_func='sum', include_friday=False, include_observed=False, binary_weekday=False):

    # Set index to pickup, sort ascending
    data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
    data = data.set_index("tpep_pickup_datetime", drop=True).sort_index()

    # Create a dictionary with all columns applying the same aggregation function
    agg_dict = {col: agg_func for col in data.columns.drop('tip_amount')}
    agg_dict['tip_amount'] = 'median'
    
    # Resample and aggregate the DataFrame
    data_time = data.resample(interval).agg(agg_dict)

    # Stitch on pickups
    pickups = data.resample(interval).size()
    data_time['pickup_count'] = pickups

    # Include weekdays
    if binary_weekday:
        if include_friday:
            data_time['weekday'] = data_time.index.weekday >= 4 
        else:
            data_time['weekday'] = data_time.index.weekday >= 5 # Doesn't include Friday
    else:
        data_time['weekday'] = data_time.index.weekday

    # Include holidays
    us_holidays = holidays.US(years=range(2022, 2025)) # 2022 to 2024
    us_holidays.observed = include_observed # decide weird observed (day falls on Sunday, so kick it to Monday)

    print("See holiday choices...")
    for date, name in sorted(us_holidays.items()):
        print(date, name)

    # Extract holidays
    holidays_list = [x[0] for x in us_holidays.items()]
    data_time['is_holiday'] = [x.date() in holidays_list for x in data_time.index]

    # Wipe data for memory
    del data
    
    return data_time

## Let's run it
if __name__ == "__main__":

    # Specify folders with data
    ## Put your parquet files here
    DATA_DIR = "raw_data"

    # Your final .csv will be here
    PROC_DATA_DIR = "processed_data"

    # Pick columns
    COLS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
       'passenger_count', 'trip_distance',
       'PULocationID', 'DOLocationID', 'fare_amount', 
       'tip_amount', 'total_amount']
    
    # Pick zone
    ZONE = 'Penn Station/Madison Sq West'

    # Pick interval for binning
    BINS = '30min'

    # Decide to include fridays
    INCLUDE_FRIDAY = False

    # Decide to include observed holidays (if falls on Sun, include Mon)
    INCLUDE_OBSERVED = False

    # Decide whether to discretize on weekend
    DISCRETIZE_WEEKEND = False

    # Get data object
    data = list()

    # Retrieve from file system
    for filename in os.listdir(DATA_DIR):

        filepath = os.path.join(DATA_DIR, filename)

        # Process the data
        zones, _ = pick_zones(zone=ZONE)

        chunk = process_data(filepath, 
                             cols_to_keep=COLS, 
                             location_ids=zones)
        
        chunk = aggregate_time_series(data=chunk,
                                     interval=BINS,
                                     agg_func='sum', # sum for aggregation over period (this applies to all columns for now)
                                     include_friday=INCLUDE_FRIDAY, # do not count friday as a weekend
                                     include_observed=INCLUDE_OBSERVED,  # do not include observed holidays
                                     binary_weekday=DISCRETIZE_WEEKEND) # discretize weekends
    
        data.append(chunk)
    
    full_data = pd.concat(data)

    # Sort and return
    full_data = full_data.sort_index()

    train_data = full_data.loc['2022':'2023']  # Include data for 2022 and 2023
    test_data = full_data.loc['2024'] 

    # Get train and test
    train_data.to_csv(os.path.join(PROC_DATA_DIR, "train.csv"))
    test_data.to_csv(os.path.join(PROC_DATA_DIR, "test.csv"))







