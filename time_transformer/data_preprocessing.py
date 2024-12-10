import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler


class PickupDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx: idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len - 1]  # Predict next step
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


def load_data(train_path, test_path, seq_len, batch_size):
    # Load train and test datasets
    train_df = pd.read_csv(train_path, parse_dates=["tpep_pickup_datetime", "pickup_time"])
    test_df = pd.read_csv(test_path, parse_dates=["tpep_pickup_datetime", "pickup_time"])

    # Features
    feature_columns = ["passenger_count", "trip_distance", "fare_amount", "total_amount", 
                       "tip_amount", "is_holiday", "weekday"]
    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values

    # Target
    y_train = train_df["pickup_count"].values
    y_test = test_df["pickup_count"].values

    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = PickupDataset(X_train, y_train, seq_len)
    test_dataset = PickupDataset(X_test, y_test, seq_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler
