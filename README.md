# STOCK-PRICE-PREDICTOR
Stock price predictor using LSTM networks

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) networks. The model is trained using historical stock data and can be used to predict future stock prices.


🚀 Setup & Installation
1️⃣ Install Required Libraries
Ensure you have the necessary dependencies installed:

pip install numpy pandas yfinance matplotlib tensorflow scikit-learn joblib
2️⃣ Download Stock Data & Train Model
Run main.ipynb to:

Fetch historical stock price data using yfinance
Preprocess the data using MinMaxScaler
Train an LSTM model
Save the trained model and scaler
🏋️‍♂️ Training the Model
The training script in main.ipynb:

Uses yfinance to fetch stock data
Normalizes data with MinMaxScaler
Defines and trains an LSTM model
Saves the trained model (.h5) and scaler (.pkl)
🧪 Testing the Model
Run test.ipynb to:

Load the saved LSTM model
Fetch the latest stock data
Predict the next closing price
Visualize actual vs. predicted prices
📊 Model Evaluation
The model is evaluated using:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score (closer to 1 is better)
📌 Example Usage
Training (main.ipynb)
# Train and save model
model = build_lstm_model(input_shape)
model.fit(X_train, y_train, epochs=20, batch_size=32)
model.save("model/lstm_model.h5")
joblib.dump(scaler, "model/scaler.pkl")
Testing (test.ipynb)
# Load and predict
model = load_model("model/lstm_model.h5")
scaler = joblib.load("model/scaler.pkl")
predicted_price = predict_stock_price("AAPL", df)
🎯 Future Enhancements
Improve model accuracy with hyperparameter tuning
Integrate real-time stock data API
Deploy as a web app for live predictions
🤝 Contributions
Feel free to fork, modify, and contribute to this project!
