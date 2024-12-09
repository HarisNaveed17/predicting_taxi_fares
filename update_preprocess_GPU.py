import os
import cudf  # GPU-accelerated DataFrame library from RAPIDS
import cupy as cp  # For GPU-based operations
import holidays
from datetime import datetime

# Specify neighborhood
def pick_zones(zone=None, borough="Manhattan"):
    try:
        zones = cudf.read_csv("taxi_zone_lookup.csv")  # Replace pandas with cudf for GPU acceleration
    except:
        print("Upload the taxi zones .csv in the directory of this script")

    if borough == "Manhattan":
        zones = zones[zones['Borough'] == 'Manhattan']
        print("See Manhattan borough options...")
        print(zones['Zone'].unique().to_pandas())  # Convert to pandas for printing if necessary

        manhattan_ids = zones['LocationID'].unique()
    else:
        print("WARNING: outside Manhattan. See borough options...")
        print(zones['Zone'].unique().to_pandas())

    if zone:
        neighborhood = zones['Zone'].str.contains(zone)
        results = zones[neighborhood]
        print("See filtered data...")
        print(results)

        location_ids = results['LocationID'].values
        print("See location ids...")
        print(location_ids)
    else:
        print("WARNING failed zone filter")

    return location_ids, manhattan_ids

# Process data
def process_data(file=None, cols_to_keep=None, location_ids=None, manhattan=False, manhattan_ids=None):
    print("Chunk: ", file)
    try:
        data = cudf.read_parquet(file)  # Use cudf for GPU-accelerated parquet reading
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return None

    if cols_to_keep:
        data = data[cols_to_keep]
    else:
        print("WARNING, keeping all data columns. Watch data size")

    # Define semester start and end dates
    start_dates = ['2022-01-24', '2023-01-23', '2024-01-22']
    end_dates = ['2022-05-17', '2023-05-16', '2024-05-14']

    # Convert dates to cudf datetime format
    start_dates = cudf.to_datetime(start_dates)
    end_dates = cudf.to_datetime(end_dates)

    # Create a mask for filtering rows based on semester dates
    mask = cp.zeros(len(data), dtype=bool)  # Use cupy for GPU-based boolean mask

    # Convert start and end dates to pandas for iteration
    start_dates_host = start_dates.to_pandas()
    end_dates_host = end_dates.to_pandas()

    # Apply the mask for each date range
    for start, end in zip(start_dates_host, end_dates_host):
        mask |= (data['tpep_pickup_datetime'] >= start) & (data['tpep_pickup_datetime'] <= end)

    # Filter data using the mask
    data = data[mask]

    # Filter by location IDs
    if len(location_ids) < 2:
        location_id_scalar = int(location_ids[0].get())  # Explicitly convert to a Python scalar
        data = data[data["PULocationID"] == location_id_scalar].drop(columns=["PULocationID"])
    else:
        location_ids_host = location_ids.to_array()  # Convert to a NumPy array on the host
        data = data[data["PULocationID"].isin(location_ids_host)].drop(columns=["PULocationID"])

    # Optionally filter by Manhattan drop-off locations
    if manhattan:
        manhattan_ids_host = manhattan_ids.to_array()  # Convert to a NumPy array on the host
        data = data[data['DOLocationID'].isin(manhattan_ids_host)]
    else:
        print("WARNING: not filtering Manhattan drop-offs...")

    # Drop unnecessary columns
    data = data.drop('tpep_dropoff_datetime', axis=1)

    return data

# Aggregate time series
def aggregate_time_series(data=None, interval='30min', agg_func='sum', include_friday=False, include_observed=False, binary_weekday=False):
    # Set index to pickup datetime and sort
    data["tpep_pickup_datetime"] = cudf.to_datetime(data["tpep_pickup_datetime"])
    data = data.set_index("tpep_pickup_datetime").sort_index()

    # Create aggregation dictionary
    agg_dict = {col: agg_func for col in data.columns.drop('tip_amount')}
    agg_dict['tip_amount'] = 'median'

    # Resample and aggregate
    data_time = data.resample(interval).agg(agg_dict)

    # Add pickup counts
    pickups = data.resample(interval).size()
    data_time['pickup_count'] = pickups

    # Add binary weekday column
    if binary_weekday:
        if include_friday:
            data_time['weekday'] = (data_time.index.weekday >= 4).astype(int)
        else:
            data_time['weekday'] = (data_time.index.weekday >= 5).astype(int)
    else:
        data_time['weekday'] = data_time.index.weekday

    # Add holiday information
    us_holidays = holidays.US(years=range(2022, 2025))
    us_holidays.observed = include_observed

    # Convert holidays to a set for faster lookup
    holidays_set = set(us_holidays.keys())

    # Create 'is_holiday' column as a cuDF Series aligned with the DataFrame index
    is_holiday_col = [date.date() in holidays_set for date in data_time.index.to_pandas()]
    is_holiday_col_gpu = cudf.Series(is_holiday_col, index=data_time.index)  # Align with cuDF DataFrame index

    # Add 'is_holiday' column to the DataFrame
    data_time['is_holiday'] = is_holiday_col_gpu

    return data_time
    
if __name__ == "__main__":
    
    DATA_DIR = "/path/to/Data"
    PROC_DATA_DIR = "/path/to/output"

    COLS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance',
            'PULocationID', 'DOLocationID', 'fare_amount', 
            'tip_amount', 'total_amount']

    ZONE = 'Penn Station/Madison Sq West'
    
    BINS = '30min'
    
    INCLUDE_FRIDAY = False
    INCLUDE_OBSERVED = False
    DISCRETIZE_WEEKEND = False

    chunks_list = []

    for filename in os.listdir(DATA_DIR):
        
        filepath = os.path.join(DATA_DIR, filename)
        
        zones, _ = pick_zones(zone=ZONE)
        
        chunk_data_gpu = process_data(filepath,
                                      cols_to_keep=COLS,
                                      location_ids=zones)
        
        chunk_agg_gpu = aggregate_time_series(data=chunk_data_gpu,
                                              interval=BINS,
                                              agg_func='sum',
                                              include_friday=INCLUDE_FRIDAY,
                                              include_observed=INCLUDE_OBSERVED,
                                              binary_weekday=DISCRETIZE_WEEKEND)
        
        chunks_list.append(chunk_agg_gpu)

    
    full_data_gpu_df = cudf.concat(chunks_list)
    # Ensure the index is sorted before slicing
    full_data_gpu_df = full_data_gpu_df.sort_index()

    # Convert index to datetime if not already
    full_data_gpu_df.index = cudf.to_datetime(full_data_gpu_df.index)

    # Filter rows for training data (2022 and 2023)
    train_mask = (full_data_gpu_df.index >= '2022-01-01') & (full_data_gpu_df.index <= '2023-12-31')
    train_data_gpu_df = full_data_gpu_df[train_mask]

    # Filter rows for testing data (2024)
    test_mask = (full_data_gpu_df.index >= '2024-01-01') & (full_data_gpu_df.index <= '2024-12-31')
    test_data_gpu_df = full_data_gpu_df[test_mask]

    # Save train and test datasets to CSV files
    train_data_gpu_df.to_csv(os.path.join(PROC_DATA_DIR, "train.csv"))
    test_data_gpu_df.to_csv(os.path.join(PROC_DATA_DIR, "test.csv"))

    
    # train_data_gpu_df = full_data_gpu_df.loc['2022':'2023']
    