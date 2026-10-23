"""Load NAB time series and attach labels from the anomaly windows file."""
import json
from pathlib import Path

import pandas as pd

NAB_DIR = Path("data/NAB")

SERIES = {
    "machine_temp": "realKnownCause/machine_temperature_system_failure.csv",
    "ec2_cpu": "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "ec2_network_in": "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "elb_requests": "realAWSCloudwatch/elb_request_count_8c0756.csv",
}


def load_windows():
    """Return {series_key: [[start, end], ...]} of labelled anomaly windows."""
    with open(NAB_DIR / "labels" / "combined_windows.json") as f:
        return json.load(f)


def load_series(key):
    """Load one series as a DataFrame: timestamp, value, is_anomaly (1 inside a labelled window)."""
    df = pd.read_csv(NAB_DIR / "data" / key, parse_dates=["timestamp"])
    df["is_anomaly"] = 0
    for start, end in load_windows()[key]:
        inside = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
        df.loc[inside, "is_anomaly"] = 1
    return df
