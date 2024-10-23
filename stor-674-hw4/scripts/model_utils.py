from sklearn import neighbors

def train_knn_regressor(x_train: pd.Series, 
                        y_train: pd.Series, 
                        n_neighbors: int, 
                        weights: str)->neighbors.KNeighborsRegressor:
    """A function wrapper to the scikit-learn API that performs fit and predict"""
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    knn.fit(x_train, y_train)
    return knn



import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder

class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """Trainin data object for our nyc taxi data"""
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = torch.from_numpy(self._one_hot_X().toarray()) # potentially smarter ways to deal with sparse here
        self.y = torch.from_numpy(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]
        print(f"encoded shape is {self.X_enc_shape}")
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """Multilayer Perceptron for regression. """
    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        return self.layers(x)

def main():
    """Simple training loop"""
    # Set fixed random number seed
    torch.manual_seed(42)
  
    # load data
    raw_df = raw_taxi_df(filename="yellow_tripdata_2024-01.parquet")
    clean_df = clean_taxi_df(raw_df=raw_df)
    location_ids = ['PULocationID', 'DOLocationID']
    X_train, X_test, y_train, y_test = split_taxi_data(clean_df=clean_df, 
                                                   x_columns=location_ids, 
                                                   y_column="fare_amount", 
                                                   train_size=500000)

    # Pytorch
    dataset = NYCTaxiExampleDataset(X_train=X_train, y_train=y_train)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    mlp = MLP(encoded_shape=dataset.X_enc_shape)
  
    # Define the loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0
    # Process is complete.
    print('Training process has finished.')
    return X_train, X_test, y_train, y_test, data, mlp

    