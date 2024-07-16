import os
import pandas as pd
import pickle
from datetime import datetime
from zipfile import ZipFile
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path, binance_data_path, training_price_data_path

def download_data():
    cm_or_um = "um"
    symbols = ["ETHUSDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path
    download_binance_monthly_data(
        cm_or_um, symbols, intervals, years, months, download_path
    )
    print(f"Downloaded monthly data to {download_path}.")
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(
        cm_or_um, symbols, intervals, current_year, current_month, download_path
    )
    print(f"Downloaded daily data to {download_path}.")

def format_data():
    files = sorted([x for x in os.listdir(binance_data_path)])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    for file in files:
        zip_file_path = os.path.join(binance_data_path, file)

        if not zip_file_path.endswith(".zip"):
            continue

        myzip = ZipFile(zip_file_path)
        with myzip.open(myzip.filelist[0]) as f:
            line = f.readline()
            header = 0 if line.decode("utf-8").startswith("open_time") else None
        df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
        df.columns = [
            "start_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "end_time",
            "volume_usd",
            "n_trades",
            "taker_volume",
            "taker_volume_usd",
        ]
        df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
        df.index.name = "date"
        price_df = pd.concat([price_df, df])

    price_df.sort_index().to_csv(training_price_data_path)

def train_model():
    # Load the eth price data
    price_data = pd.read_csv(training_price_data_path)

    # Convert to long format for AutoGluon
    price_data = price_data[["date", "close"]].rename(columns={"date": "timestamp", "close": "target"})
    price_data["item_id"] = "ETHUSDT"
    price_data["timestamp"] = pd.to_datetime(price_data["timestamp"])

    train_data = TimeSeriesDataFrame.from_data_frame(
        price_data,
        id_column="item_id",
        timestamp_column="timestamp"
    )

    predictor = TimeSeriesPredictor(
        prediction_length=48,
        path="autogluon-m4-hourly",
        target="target",
        eval_metric="MASE",
    )

    predictor.fit(
        train_data,
        presets="medium_quality",
        time_limit=600,
    )

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(predictor, f)

    print(f"Trained model saved to {model_file_path}")
