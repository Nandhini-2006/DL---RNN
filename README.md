# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: NANDHINI N

### Register Number: 212224040212

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockPricePredictor:
    def __init__(self, symbol='AAPL', lookback_days=60):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_data(self, start_date='2015-01-01', end_date='2023-12-31'):
        """Fetch historical stock data"""
        print(f"Fetching data for {self.symbol}...")
        self.stock_data = yf.download(self.symbol, start=start_date, end=end_date)
        return self.stock_data
    
    def prepare_data(self, train_split=0.8):
        """Prepare data for training and testing"""
        # Use closing prices
        closing_prices = self.stock_data['Close'].values.reshape(-1, 1)
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(closing_prices)
        
        # Create training and test sets
        train_size = int(len(scaled_data) * train_split)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)
        
        return X_train, y_train, X_test, y_test
    
    def create_sequences(self, data):
        """Create input sequences and corresponding targets"""
        X, y = [], []
        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, units=50, dropout_rate=0.2):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(units=units, return_sequences=True, 
                 input_shape=(self.lookback_days, 1)),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return self.model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train the model"""
        print("Training the model...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=1,
            shuffle=False
        )
        return history
    
    def predict(self, X_test, y_test):
        """Make predictions"""
        # Reshape for LSTM input
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Make predictions
        predictions = self.model.predict(X_test_reshaped)
        
        # Inverse transform predictions and actual values
        predictions_actual = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        return predictions_actual, y_test_actual
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"\nModel Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return mse, rmse, mae, mape
    
    def plot_results(self, y_true, y_pred, history):
        """Plot training history and predictions"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training history
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Training History')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot predictions vs actual
        ax2.plot(y_true, label='Actual Prices', alpha=0.7)
        ax2.plot(y_pred, label='Predicted Prices', alpha=0.7)
        ax2.set_title(f'{self.symbol} Stock Price Prediction')
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        # Get the last sequence from available data
        closing_prices = self.stock_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(closing_prices)
        
        last_sequence = scaled_data[-self.lookback_days:]
        future_predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape((1, self.lookback_days, 1))
            
            # Predict next day
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_prices = self.scaler.inverse_transform(future_predictions)
        
        return future_prices

def main():
    # Initialize the predictor
    predictor = StockPricePredictor(symbol='AAPL', lookback_days=60)
    
    # Fetch data
    stock_data = predictor.fetch_data(start_date='2015-01-01', end_date='2023-12-31')
    print(f"Data shape: {stock_data.shape}")
    print(f"Data columns: {stock_data.columns.tolist()}")
    
    # Prepare data
    X_train, y_train, X_test, y_test = predictor.prepare_data(train_split=0.8)
    print(f"Training data shape: X_train{X_train.shape}, y_train{y_train.shape}")
    print(f"Test data shape: X_test{X_test.shape}, y_test{y_test.shape}")
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build model
    model = predictor.build_model(units=50, dropout_rate=0.2)
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    history = predictor.train_model(X_train, y_train, epochs=50, batch_size=32)
    
    # Make predictions
    predictions, actual_values = predictor.predict(
        X_test.reshape(X_test.shape[0], X_test.shape[1]), 
        y_test
    )
    
    # Evaluate model
    mse, rmse, mae, mape = predictor.evaluate_model(actual_values, predictions)
    
    # Plot results
    predictor.plot_results(actual_values, predictions, history)
    
    # Predict future prices
    future_prices = predictor.predict_future(days=30)
    print(f"\nNext 30 days predicted prices:")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ${price[0]:.2f}")

if __name__ == "__main__":
    main()

```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1199" height="1171" alt="Screenshot 2025-11-28 195045" src="https://github.com/user-attachments/assets/d297491e-c509-4147-96c4-e88d78c20040" />

<img width="1064" height="805" alt="Screenshot 2025-11-28 195055" src="https://github.com/user-attachments/assets/3689b49d-07ba-453b-9aea-a5cec81c3070" />

<img width="1368" height="1095" alt="Screenshot 2025-11-28 195201" src="https://github.com/user-attachments/assets/4c9a1961-94af-4fee-aa4c-853929277785" />


### Predictions

<img width="455" height="550" alt="Screenshot 2025-11-28 195305" src="https://github.com/user-attachments/assets/b16056f0-ba24-4bc3-9b1a-16b038f5c9df" />


## RESULT

Hence, a Recurrent Neural Network (RNN) model is created for predicting stock prices using historical closing price data.
