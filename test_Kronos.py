from model import Kronos, KronosTokenizer, KronosPredictor
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_prediction(kline_df, pred_df):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

print(predictor)

#dir = 'D:\\git\\futu_algo\\data\\SZ.301536\\SZ.301536_2025_1D.parquet'
#df = pd.read_parquet(dir)

data = 'D:\\git\\futu_algo\\zz800.test'
df = pd.read_csv(data)
df['timestamps'] = pd.to_datetime(df['time_key'])
df['amount'] = df['turnover_lag_0']
df['volume'] = df['volume_lag_0']
df['open'] = df['open_lag_0']
df['close'] = df['close_lag_0']
df['high'] = df['high_lag_0']
df['low'] = df['low_lag_0']
df.sort_values('timestamps', inplace=True)

print(df)

group = df.groupby('code')
# Define context window and prediction length
lookback = 40
pred_len = 5

for code, df in group:
    print(f"Processing group for code: {code}")
    # Prepare inputs for the predictor
    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']
    y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

    # Generate predictions
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,          # Temperature for sampling
        top_p=0.9,      # Nucleus sampling probability
        sample_count=1  # Number of forecast paths to generate and average
    )

    print("Forecasted Data Head:")
    print(pred_df.head())

    # Combine historical and forecasted data for plotting
    kline_df = df.loc[:lookback+pred_len-1]

    # visualize
    plot_prediction(kline_df, pred_df)