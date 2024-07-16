import json
import pickle
import pandas as pd
from datetime import datetime
from flask import Flask, Response
from autogluon.timeseries import TimeSeriesDataFrame
from model import download_data, format_data, train_model
from config import model_file_path

app = Flask(__name__)

def update_data():
    """Download price data, format data, and train model."""
    download_data()
    format_data()
    train_model()

def get_eth_inference():
    """Load model and predict current price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    now_timestamp = pd.Timestamp(datetime.now()).timestamp()
    new_data = pd.DataFrame({
        "item_id": ["ETHUSDT"],
        "timestamp": [now_timestamp],
        "target": [None]  # Placeholder for target
    })

    # Convert timestamp to datetime
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='s')

    # Ensure 'timestamp' is the index
    new_data.set_index('timestamp', inplace=True)

    # Convert to TimeSeriesDataFrame
    new_ts_data = TimeSeriesDataFrame.from_data_frame(new_data.reset_index(), id_column="item_id", timestamp_column="timestamp")

    print("New TimeSeriesDataFrame for inference:", new_ts_data)  # Debugging line

    # Specify the model for predictions
    predictions = loaded_model.predict(new_ts_data)
    current_price_pred = predictions["mean"].iloc[0]

    return current_price_pred

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token != "ETH":
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_eth_inference()
        print(f"Inference value for {token}: {inference}")  # Log inference value
        return Response(str(inference), status=200)
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Log error
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception as e:
        print(f"Update error: {str(e)}")  # Log update error
        return "1"

if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000, threaded=True)
