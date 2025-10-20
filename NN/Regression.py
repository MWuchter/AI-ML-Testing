import os
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------
# 1. Configuration
DATA_DIR        = 'STOCK_DATA/Stocks'
WINDOW_SIZE     = 5            # days per sample
TRAIN_FRACTION  = 0.8          # fraction of tickers used for training
MIN_ROWS        = WINDOW_SIZE + 1  # minimum rows needed to form at least one sample
RIDGE_LAMBDA    = 1e-3         # Regularization strength

# ------------------------------------------------------------------------------------------
# 2. Helper: load one file into feature windows + targets
def load_stock(file_path, window_size):
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, names=True, encoding=None)
    if data.size < MIN_ROWS:
        return np.empty((0, window_size * 5)), np.empty((0,))
    days = np.column_stack((data['Open'], data['High'], data['Low'], data['Close'], data['Volume']))
    n_samples = len(days) - window_size
    X, y = [], []
    for i in range(n_samples):
        X.append(days[i:i+window_size].ravel())
        y.append(data['Close'][i + window_size])
    return np.array(X), np.array(y)

# ------------------------------------------------------------------------------------------
# 3. Gather ticker files with sufficient data
all_X, all_y, tickers = [], [], []

for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith('.txt'):
        continue
    path = os.path.join(DATA_DIR, fname)
    with open(path, 'r') as f:
        lines = f.readlines()
    if len(lines) < MIN_ROWS + 1:
        print(f"Skipping {fname}: not enough data (only {len(lines)-1} rows)")
        continue
    ticker = os.path.splitext(fname)[0]
    tickers.append(ticker)

# ------------------------------------------------------------------------------------------
# 4. Split into train/test
train_tickers, test_tickers = train_test_split(tickers, train_size=TRAIN_FRACTION, random_state=42)

# ------------------------------------------------------------------------------------------
# 5. Load training data
for ticker in train_tickers:
    path = os.path.join(DATA_DIR, ticker + '.txt')
    X_t, y_t = load_stock(path, WINDOW_SIZE)
    if X_t.size == 0:
        print(f"No samples for training from {ticker}, skipping.")
        continue
    all_X.append(X_t)
    all_y.append(y_t)
X_train = np.vstack(all_X)
y_train = np.hstack(all_y)

# ------------------------------------------------------------------------------------------
# 6. Load testing data
all_X, all_y = [], []
for ticker in test_tickers:
    path = os.path.join(DATA_DIR, ticker + '.txt')
    X_t, y_t = load_stock(path, WINDOW_SIZE)
    if X_t.size == 0:
        print(f"No samples for testing from {ticker}, skipping.")
        continue
    all_X.append(X_t)
    all_y.append(y_t)
X_test = np.vstack(all_X)
y_test = np.hstack(all_y)

# ------------------------------------------------------------------------------------------
# 7. Standardize features using training stats
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1  # Prevent division by zero
X_train_std = (X_train - mean) / std
X_test_std  = (X_test - mean) / std

# ------------------------------------------------------------------------------------------
# 8. Add bias column
X_train_b = np.c_[np.ones((X_train_std.shape[0], 1)), X_train_std]
X_test_b  = np.c_[np.ones((X_test_std.shape[0], 1)),  X_test_std]

# ------------------------------------------------------------------------------------------
# 9. Ridge Regression with closed-form solution (do not regularize bias)
n_features = X_train_b.shape[1]
ridge_matrix = np.eye(n_features)
ridge_matrix[0, 0] = 0  # No penalty for bias term

XtX_ridge = X_train_b.T @ X_train_b + RIDGE_LAMBDA * ridge_matrix
w = np.linalg.inv(XtX_ridge) @ (X_train_b.T @ y_train)
print("Learned weights (w) shape:", w.shape)

# ------------------------------------------------------------------------------------------
# 10. Evaluate on training set
y_train_pred = X_train_b @ w
mse_train = np.mean((y_train - y_train_pred)**2)
r2_train  = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - np.mean(y_train))**2)
print(f"Train MSE: {mse_train:.4f}, R²: {r2_train:.4f}")

# ------------------------------------------------------------------------------------------
# 11. Evaluate on testing set
y_test_pred = X_test_b @ w
mse_test = np.mean((y_test - y_test_pred)**2)
r2_test  = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
print(f"Test MSE: {mse_test:.4f}, R²: {r2_test:.4f}")
