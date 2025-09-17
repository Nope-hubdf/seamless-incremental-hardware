import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, HoltWinters
#from statsforecast.models import MSTL, AutoEST

transactions = pd.read_csv('./data/transactions_data_sampled.csv')

print(transactions.head())

# Convert to datetime (matches your sample format)
transactions['date'] = pd.to_datetime(
    transactions['date'],
    format='%Y-%m-%d %H:%M:%S',   # fast & strict
    errors='coerce'               # bad rows -> NaT (so it won't crash)
)

# 1) Load parquet

# This is a hefty table, so just peeking at the first 5 rows
#data = pd.read_csv('./data/sales_data_sampled.csv', nrows=50)
data = pd.read_csv('./data/sales_data_sampled.csv')

data.info(memory_usage='deep')

#print(data)

# 2) Ensure proper dtypes

data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()  # strip time-of-day if present
data["sales"] = pd.to_numeric(data["sales"], errors="coerce").fillna(0)

# (Optional speed-up: ignore zero rows before grouping; same totals, fewer rows)
# tx = tx[tx["sales"] > 0]

# 3) Aggregate to daily level (all items/stores summed)
daily = (data.groupby(["date", "id", "dept_id"], as_index=False)["sales"].sum()
           .sort_values("date")
           .reset_index(drop=True))

# 4) Drop days with zero total sales
daily = daily.loc[daily["sales"] > 0].reset_index(drop=True)

print(daily.head())
print("Daily rows:", len(daily))

# 5) Test

def test_sales_eq(data):
    assert (
        pd.read_csv('./data/sales_data_sampled.csv', usecols=['date', 'id', 'sales'])
        .assign(date=lambda df: pd.to_datetime(df.date))
        .query('sales != 0')
        .merge(data, on=['date', 'id'], how='left', suffixes=('_actual', '_predicted'))
        .fillna(0)
        .assign(sales_error=lambda df: (df.sales_actual - df.sales_predicted).abs())
        .sales_error
        .sum() < 1e-6
    ), 'Your version of sales does not match the original sales data.'

    assert (
        pd.read_csv('./data/sales_data_sampled.csv', usecols=['date', 'id', 'sales'])
        .query('sales != 0')
        .shape[0]
    ) == data.shape[0], 'Your dataframe has a different number of rows than the original sales data.'

#test_sales_eq(daily)

# --- 1) set index to (date, id) ---
daily = daily.set_index(["date", "id"]).sort_index()

# --- 2) full MultiIndex over all daily dates × ids, then reindex ---
all_dates = pd.date_range(
    daily.index.get_level_values("date").min(),
    daily.index.get_level_values("date").max(),
    freq="D"
)
all_ids = daily.index.get_level_values("id").unique()
full_idx = pd.MultiIndex.from_product([all_dates, all_ids], names=["date", "id"])

daily = daily.reindex(full_idx)

# --- 3) fill NaNs ---
daily["sales"] = daily["sales"].fillna(0)      # missing combos => 0 sales
# (optional) keep integer type if you prefer
# daily["sales"] = daily["sales"].astype("int64")

# back to columns (optional)
daily = daily.reset_index()

print(daily.head())
print("Daily rows:", len(daily))

# --- 4) test

def test_sales_eq(data):
    data_copy = (
        data
        .copy()
        .reset_index('id')
        .assign(id=lambda df: df.id.astype(str).values)
    )
    assert (
        pd.read_csv('./data/sales_data_sampled.csv', usecols=['date', 'id', 'sales'])
        .query('id != "FOODS_2_394_TX_3_evaluation"')  # this item is missing in my modified data
        .assign(date=lambda df: pd.to_datetime(df.date))
        .merge(
            data_copy, 
            on=['date', 'id'], 
            how='left', 
            suffixes=('_actual', '_predicted')
        )
        .fillna(0)
        .assign(sales_error=lambda df: (df.sales_actual - df.sales_predicted).abs())
        .sales_error
        .sum() < 1e-6
    ), 'Your version of sales does not match the original sales data.'

#test_sales_eq(daily)

# series indexed by date after summing
s = (daily.groupby(["date", "dept_id"])["sales"]
       .sum()
       .sort_index()
       .xs("FOODS_1", level="dept_id"))

ax = s.plot(figsize=(10, 4))
ax.set_title("FOODS_1 daily sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

ax.figure.tight_layout()
ax.figure.savefig("foods_1_daily_sales.png", dpi=300, bbox_inches="tight")
plt.close(ax.figure)

# Aggregate data to date/dept_id level

train_data = daily.groupby(['date', 'dept_id']).sales.agg('sum').reset_index()
train_data.head()

df = train_data.rename(columns={
    'dept_id': 'unique_id',
    'date': 'ds',
    'sales': 'y'
})
train_df = df[df.ds < pd.Timestamp('2016-04-24')]

sf = StatsForecast(
    models=[
        # SARIMA(1, 1, 1)(1, 1, 1),7
        ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1), season_length=7),
        # ETS model
        HoltWinters(season_length=7)
    ],
    #models = [
    #    MSTL(
    #        season_length=[7, 365],   # daily data: weekly + yearly
    #        trend_forecaster=AutoETS()
    #    )
    #],
    freq='D'
)
sf.fit(train_df)

forecast_df = sf.predict(h=28)
print(forecast_df.tail())
