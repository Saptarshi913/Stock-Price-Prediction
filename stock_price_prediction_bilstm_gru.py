

from google.colab import files
files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Step 1: Load your data
df = pd.read_csv('GE.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Define features and target
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df['MA_10'] = df['Adj Close'].rolling(window=10).mean()
df['MA_30'] = df['Adj Close'].rolling(window=30).mean()
df['MA_50'] = df['Adj Close'].rolling(window=50).mean()
feature_cols += ['MA_10', 'MA_30', 'MA_50']
target_col = 'Adj Close'

# Handle NaN values
df.dropna(subset=['MA_10', 'MA_30', 'MA_50'], inplace=True)  # Drop rows with NaN values

# Step 3: Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_X = scaler_X.fit_transform(df[feature_cols])
scaled_y = scaler_y.fit_transform(df[[target_col]])
scaled_data = np.hstack((scaled_X, scaled_y))  # Combine features and target

# Step 4: Sequence preparation
def create_sequences(data, n_steps, n_features):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

n_steps = 60
n_features = len(feature_cols)

# Step 5: Model creation function with tanh activation and units=50
def create_model(units=50, dropout_rate=0.2):
    input_layer = Input(shape=(n_steps, n_features))
    x = Bidirectional(LSTM(units, activation='tanh', return_sequences=True))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(units, activation='tanh', return_sequences=True))(x)  # Using GRU
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(units, activation='tanh'))(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')
    return model

# Step 6: Cross-validation using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
mape_scores = []

for train_index, test_index in tscv.split(scaled_data):
    X_train = scaled_data[train_index]
    X_test = scaled_data[test_index]

    X_train_seq, y_train_seq = create_sequences(X_train, n_steps, n_features)
    X_test_seq, y_test_seq = create_sequences(X_test, n_steps, n_features)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Fit the model
    model = create_model(units=50)  # Use 50 units as specified
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])

    y_pred = model.predict(X_test_seq)
    if np.isnan(y_pred).any():  # Check if any NaN values in predictions
        print("Warning: NaN values found in predictions!")
        continue
    mape = mean_absolute_percentage_error(y_test_seq, y_pred)
    mape_scores.append(mape)

# Step 7: Evaluate
average_mape = np.mean(mape_scores)
print(f'\n✅ Average MAPE from Cross-Validation: {average_mape:.4f}')

# Step 8: Final training on full data
X_full, y_full = create_sequences(scaled_data, n_steps, n_features)

model = create_model(units=50)  # Use 50 units as specified
model.fit(X_full, y_full, epochs=50, batch_size=32, verbose=1)  # 50 epochs here

# Step 9: Prediction on last portion
test_data = scaled_data[-(len(y_test_seq) + n_steps):]
X_test_final, y_test_final = create_sequences(test_data, n_steps, n_features)

y_pred_final = model.predict(X_test_final)
if np.isnan(y_pred_final).any():
    print("Warning: NaN values found in final predictions!")
y_pred_rescaled = scaler_y.inverse_transform(y_pred_final)
y_test_rescaled = scaler_y.inverse_transform(y_test_final.reshape(-1, 1))

# Step 10: Plot results
test_dates = df.index[-len(y_test_final):]
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test_rescaled, color='red', label='Actual Adj Close')
plt.plot(test_dates, y_pred_rescaled, color='green', label='Predicted Adj Close')
plt.title('Bidirectional LSTM-GRU Model — Adj Close Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 11: Final evaluation metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"MAE  (Mean Absolute Error):      {mae:.3f}")
print(f"MSE  (Mean Squared Error):       {mse:.3f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.3f}")
print(f"R²   (Coefficient of Determination): {r2:.3f}")

