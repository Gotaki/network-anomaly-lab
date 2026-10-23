"""Summarise and plot the chosen NAB series with labelled anomaly windows shaded."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load import SERIES, load_series, load_windows

Path("reports/figures").mkdir(parents=True, exist_ok=True)

rows = []
fig, axes = plt.subplots(len(SERIES), 1, figsize=(13, 2.8 * len(SERIES)))
for ax, (name, key) in zip(axes, SERIES.items()):
    df = load_series(key)
    step = df["timestamp"].diff().mode()[0]
    rows.append({
        "series": name,
        "rows": len(df),
        "start": df["timestamp"].min().date(),
        "end": df["timestamp"].max().date(),
        "interval": str(step),
        "windows": len(load_windows()[key]),
        "anomaly_rows_%": round(100 * df["is_anomaly"].mean(), 1),
    })
    ax.plot(df["timestamp"], df["value"], lw=0.5)
    ax.fill_between(df["timestamp"], df["value"].min(), df["value"].max(),
                    where=df["is_anomaly"] == 1, color="red", alpha=0.25)
    ax.set_title(name, fontsize=9, loc="left")

print(pd.DataFrame(rows).to_string(index=False))
plt.tight_layout()
plt.savefig("reports/figures/01_series.png", dpi=110)
print("saved reports/figures/01_series.png")
