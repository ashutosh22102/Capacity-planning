# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Load Data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Streamlit UI
st.title("Building Used Ports Prediction Dashboard")

# File upload
file_path = st.text_input("Enter the path to your dataset file:", value="Data.csv")
data = load_data(file_path)

# Select Building ID
building_ids = data['Building_ID'].unique()
selected_building_id = st.selectbox("Select Building ID:", building_ids)

# Filter data for selected Building ID
building_data = data[data['Building_ID'] == selected_building_id]

# Extract columns with 'Used Ports'
used_ports_cols = [col for col in building_data.columns if 'Used Ports' in col]

# Reshape and process time data
time_data = building_data[['Building_ID'] + used_ports_cols].melt(
    id_vars=['Building_ID'],
    var_name='Date',
    value_name='Used Ports'
)
time_data['Date'] = time_data['Date'].str.split('_').str[0]
time_data['Date'] = pd.to_datetime(time_data['Date'], errors='coerce')
time_data = time_data.dropna(subset=['Date']).sort_values('Date')

# Add date range selection
st.subheader("Select Date Range")
min_date, max_date = time_data['Date'].min(), time_data['Date'].max()
start_date, end_date = st.date_input("Date range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter data by selected date range
filtered_data = time_data[(time_data['Date'] >= pd.Timestamp(start_date)) & (time_data['Date'] <= pd.Timestamp(end_date))]

# Plot actual data
st.subheader(f"Used Ports Over Time for {selected_building_id}")
st.line_chart(data=filtered_data.set_index('Date')['Used Ports'])

# Normalize data and create sequences
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_data[['Used Ports']])

sequence_length = 30
X, y = create_sequences(scaled_data, sequence_length)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build and Train LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='loss', patience=10)

# Train model
model.fit(X, y, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0)

# Predict historical data
historical_predictions = model.predict(X)
historical_predictions = scaler.inverse_transform(historical_predictions).flatten()
historical_predictions = np.round(historical_predictions).astype(int)  # Ensure integer values

# Predict future data
st.subheader("Future Predictions")
future_days = st.slider("Select number of future days to predict:", min_value=1, max_value=365, value=180)

predicted_used_ports = []
last_30_days = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

for _ in range(future_days):
    next_pred = model.predict(last_30_days)
    predicted_used_ports.append(next_pred[0, 0])
    last_30_days = np.append(last_30_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

predicted_used_ports = scaler.inverse_transform(np.array(predicted_used_ports).reshape(-1, 1)).flatten()
predicted_used_ports = np.round(predicted_used_ports).astype(int)  # Ensure integer values

# Future predictions dataframe
last_date = time_data['Date'].iloc[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1, freq='D')[1:]
future_df = pd.DataFrame({'Date': future_dates, 'Forecast': predicted_used_ports})

# Combine historical and forecast data for interactive plot
historical_df = time_data[['Date', 'Used Ports']]
historical_df['Predicted'] = pd.Series(historical_predictions, index=historical_df.index[-len(historical_predictions):])
combined_df = pd.concat([historical_df, future_df.set_index('Date')], axis=0)

# Plot interactive graph
st.subheader("Interactive Plot")
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Used Ports'],
                         mode='lines', name='Actual Data', line=dict(color='blue')))

# Add predicted data
fig.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Predicted'],
                         mode='lines', name='Predicted Data', line=dict(color='green')))

# Add forecast data
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Forecast'],
                         mode='lines', name='Forecast Data', line=dict(color='red')))

# Customize layout
fig.update_layout(
    title=f"Used Ports Analysis for {selected_building_id}",
    xaxis_title="Date",
    yaxis_title="Used Ports",
    legend_title="Legend",
    template="plotly_white"
)

# Display interactive plot
st.plotly_chart(fig)

# Download button for predictions
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions", data=csv, file_name=f"future_predictions_{selected_building_id}.csv", mime="text/csv")
