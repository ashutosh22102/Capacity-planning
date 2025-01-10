from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import io

app = Flask(__name__)

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Load Data
def load_data():
    file_path = 'Data.csv'
    data = pd.read_csv(file_path)
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    data = load_data()
    building_ids = data['Building_ID'].unique()
    graph = None

    if request.method == 'POST':
        selected_building_id = request.form.get('building_id')
        future_days = int(request.form.get('future_days', 180))

        building_data = data[data['Building_ID'] == selected_building_id]
        used_ports_cols = [col for col in building_data.columns if 'Used Ports' in col]

        time_data = building_data[['Building_ID'] + used_ports_cols].melt(
            id_vars=['Building_ID'],
            var_name='Date',
            value_name='Used Ports'
        )
        time_data['Date'] = pd.to_datetime(time_data['Date'].str.split('_').str[0], errors='coerce')
        time_data = time_data.dropna(subset=['Date']).sort_values('Date')

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(time_data[['Used Ports']])

        sequence_length = 30
        X, y = create_sequences(scaled_data, sequence_length)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='loss', patience=10)
        model.fit(X, y, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0)

        predicted_used_ports = []
        last_30_days = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

        for _ in range(future_days):
            next_pred = model.predict(last_30_days)
            predicted_used_ports.append(next_pred[0, 0])
            last_30_days = np.append(last_30_days[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

        predicted_used_ports = scaler.inverse_transform(np.array(predicted_used_ports).reshape(-1, 1)).flatten()
        predicted_used_ports = np.round(predicted_used_ports).astype(int)

        last_date = time_data['Date'].iloc[-1]
        future_dates = pd.date_range(last_date, periods=future_days + 1, freq='D')[1:]
        future_df = pd.DataFrame({'Date': future_dates, 'Forecast': predicted_used_ports})

        historical_predictions = model.predict(X)
        historical_predictions = scaler.inverse_transform(historical_predictions).flatten()
        historical_predictions = np.round(historical_predictions).astype(int)

        historical_df = time_data[['Date', 'Used Ports']].copy()
        historical_df['Predicted'] = pd.Series(historical_predictions, index=historical_df.index[-len(historical_predictions):])
        combined_df = pd.concat([historical_df, future_df.set_index('Date')], axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Used Ports'],
                                 mode='lines', name='Actual Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Predicted'],
                                 mode='lines', name='Predicted Data', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Forecast'],
                                 mode='lines', name='Forecast Data', line=dict(color='red')))
        fig.update_layout(
            title=f"Used Ports Analysis for {selected_building_id}",
            xaxis_title="Date",
            yaxis_title="Used Ports",
            legend_title="Legend",
            template="plotly_white"
        )
        graph = fig.to_html(full_html=False)

    return render_template('index.html', building_ids=building_ids, graph=graph)

if __name__ == '__main__':
    app.run(debug=True)