# Capacity Planning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

An intelligent AI-driven solution for predicting network port usage and capacity planning using LSTM neural networks. The system analyzes historical port usage data to forecast future requirements, helping organizations optimize their network infrastructure.

## Features

- **AI-Powered Forecasting:** 
    - LSTM-based time series prediction
    - Historical pattern analysis
    - Adaptive learning capabilities
- **Interactive Visualization:** 
    - Real-time graphical predictions
    - Historical vs predicted comparisons
    - Dynamic date range selection
- **Dual Interface Options:**
    - Web-based Flask interface
    - Streamlit dashboard
- **Enterprise Ready:** 
    - Multi-building support
    - Scalable architecture
    - CSV data integration

## Project Structure

```
├── app.py              # Flask web application
├── gui_code.py         # Streamlit dashboard
├── requirements.txt    # Project dependencies
├── Data.csv           # Historical port usage data
├── templates/         
│   └── index.html     # Flask web interface
└── venv/              # Virtual environment
```

## Requirements

```python
# Core Dependencies
flask>=3.1.0
streamlit>=1.41.1
tensorflow>=2.18.0
pandas>=2.2.3
numpy>=2.0.2
scikit-learn>=1.6.0
plotly>=5.24.1
```

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/capacity-planning.git
cd capacity-planning
```

2. **Set Up Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Flask Web Interface
```bash
python app.py
# Access at http://localhost:5000
```

### Streamlit Dashboard
```bash
streamlit run gui_code.py
# Access at http://localhost:8501
```

## Features Detail

### Data Processing
- Time series data normalization
- Sequence generation for LSTM
- MinMaxScaler implementation

### Model Architecture
- LSTM neural network
- Dense output layer
- Early stopping implementation

### Visualization
- Interactive Plotly graphs
- Historical data overlay
- Future predictions display

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for LSTM implementation
- Streamlit for interactive dashboard capabilities
- Flask for web framework support
