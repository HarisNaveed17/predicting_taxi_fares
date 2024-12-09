import cudf  # GPU-accelerated DataFrame library
import cupy as cp  # GPU-based numerical computations
import torch  # PyTorch for GPU-accelerated tensor operations
import gpytorch  # GPyTorch for Gaussian Process Regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training data using cuDF
train = cudf.read_csv('/path/to/train.csv')
train['tip_amount'] = train['tip_amount'].fillna(0)
train['pickup_count'] = train['pickup_count'].fillna(0)  # Ensure no nulls in target column
train['tpep_pickup_datetime'] = cudf.to_datetime(train['tpep_pickup_datetime'])
train = train.set_index('tpep_pickup_datetime')

# Add interval and time index columns
train['interval'] = cp.arange(1, len(train) + 1) % 48  # Get to hour interval
train['time_index'] = cp.arange(1, len(train) + 1)

# Limit to first 800 rows for training (convert to pandas for preview if needed)
# train = train.iloc[:800]
print(train.head().to_pandas())  # Convert to pandas for display

# Load testing data using cuDF
test = cudf.read_csv('/path/to/test.csv')
test['tip_amount'] = test['tip_amount'].fillna(0)
test['pickup_count'] = test['pickup_count'].fillna(0)  # Ensure no nulls in target column
test['tpep_pickup_datetime'] = cudf.to_datetime(test['tpep_pickup_datetime'])
test = test.set_index('tpep_pickup_datetime')

# Add interval and time index columns
test['interval'] = cp.arange(1, len(test) + 1) % 48  # Get to hour interval
test['time_index'] = cp.arange(1, len(test) + 1)

# Limit to first 800 rows for testing (convert to pandas for preview if needed)
# test = test.iloc[:800]
print(test.head().to_pandas())  # Convert to pandas for display

# Features (X) - All columns except 'pickup_count'
X_train = train[['interval', 'weekday']].to_cupy()  # Convert cuDF to CuPy array
X_test = test[['interval', 'weekday']].to_cupy()    # Convert cuDF to CuPy array

# Target (y) - 'pickup_count'
mean_func = train['pickup_count'].mean()
y_train = train['pickup_count'].to_cupy() - mean_func  # Center target variable
y_test = test['pickup_count'].to_cupy() - mean_func

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(cp.asnumpy(X_train))  # Convert CuPy array to NumPy for scaling
X_test_scaled = scaler.transform(cp.asnumpy(X_test))       # Convert CuPy array to NumPy for scaling

# Convert scaled features back to PyTorch tensors for GPU processing
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define the GP model using GPyTorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + 
            gpytorch.kernels.PeriodicKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = ExactGPModel(X_train_torch, y_train_torch, likelihood).to(device)

# Set the model in training mode
model.train()
likelihood.train()

# Use the Adam optimizer (gradient-based optimization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Loss function: Marginal Log Likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
num_iterations = 50  # Number of optimization iterations
for i in range(num_iterations):
    optimizer.zero_grad()
    output = model(X_train_torch)
    loss = -mll(output, y_train_torch)  # Negative log likelihood
    loss.backward()
    print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss.item():.3f}")
    optimizer.step()

# Set the model in evaluation mode for predictions
model.eval()
likelihood.eval()

# Make predictions on test data
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(X_test_torch))
    mean_pred = predictions.mean.cpu().numpy()  # Move results back to CPU for evaluation
    lower, upper = predictions.confidence_region()
    lower = lower.cpu().numpy()
    upper = upper.cpu().numpy()

# Evaluate the model's performance
rmse = mean_squared_error(cp.asnumpy(y_test), mean_pred, squared=False)
r2 = r2_score(cp.asnumpy(y_test), mean_pred)
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# Visualization of predictions with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), cp.asnumpy(y_test), label="True Values", color="blue", alpha=0.6)
plt.plot(range(len(mean_pred)), mean_pred, label="GP Predictions", color="red", linestyle="dashed")
plt.fill_between(
    range(len(mean_pred)),
    lower,
    upper,
    color="lightgrey",
    alpha=0.5,
    label="95% Confidence Interval",
)
plt.xlabel("Sample Index")
plt.ylabel("Pickup Count")
plt.title("Gaussian Process Regression: Pickup Count Prediction")
plt.legend()
plt.show()