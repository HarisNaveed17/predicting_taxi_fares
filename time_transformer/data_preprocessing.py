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


def load_data(file_path, batch_size, seq_len):
    # Load dataset
    df = pd.read_csv(file_path, parse_dates=["tpep_pickup_datetime", "pickup_time"])

    # Features
    features = df[["passenger_count", "trip_distance", "fare_amount", "total_amount", 
                   "tip_amount", "is_holiday", "weekday"]].values

    # Target
    target = df["pickup_count"].values

    # Scale features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Dataset
    dataset = PickupDataset(features, target, seq_len)

    # Train-test split
    split_idx = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler
