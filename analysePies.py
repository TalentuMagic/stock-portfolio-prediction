import numpy as np
import pandas as pd
from piesHistroricalData import OwnTheWorldIn50, ETFs, EurozoneInvestments, Crypto
import matplotlib.pyplot as plt


def dataSetup(pieData: list = None):
    for index, stock_data in enumerate(pieData):
        stock_data['RSI'] = computeRSI(stock_data=stock_data)
        stock_data['ATR'] = computeATR(stock_data=stock_data)
        stock_data['WMA'] = computeWMA(stock_data['Adj Close'], 14)
        stock_data['MACD'], stock_data['MACD_Signal'] = computeMACD(
            stock_data['Adj Close'], 12, 26)
        plus_di, minus_di, adx_smooth = computeDMI_ADX(
            stock_data['High'], stock_data['Low'], stock_data['Close'], 14)
        stock_data['PlusDI'] = plus_di
        stock_data['MinusDI'] = minus_di
        stock_data['ADX'] = adx_smooth
        stock_data['SMA_50'] = stock_data['Adj Close'].rolling(
            window=50).mean()
        stock_data['EMA_50'] = stock_data['Adj Close'].ewm(
            span=50, adjust=False).mean()
        stock_data['SMA_20'] = stock_data['Adj Close'].rolling(
            window=20).mean()
        stock_data['UpperBand'] = stock_data['SMA_20'] + \
            2 * stock_data['Adj Close'].rolling(window=20).std()
        stock_data['LowerBand'] = stock_data['SMA_20'] - \
            2 * stock_data['Adj Close'].rolling(window=20).std()
        stock_data['Tomorrow'] = stock_data['Adj Close'].shift(-1)
        stock_data['ROI'] = ((stock_data['Adj Close'].shift(
            -1) - stock_data['Adj Close']) / stock_data['Adj Close'])*100
        stock_data['Target'] = stock_data['Adj Close'].shift(
            -1) - stock_data['Adj Close']
        stock_data['TargetClass'] = (
            stock_data['Tomorrow'] > stock_data['Adj Close']).astype(int)

        pieData[index] = pieData[index].dropna()

    return pieData


def computeRSI(stock_data: pd.DataFrame):
    change = stock_data["Adj Close"].diff()
    # Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()

    #
    change_up[change_up < 0] = 0
    change_down[change_down > 0] = 0

    # Verify that we did not make any mistakes
    stock_data.equals(change_up+change_down)

    # Calculate the rolling average of average up and average down
    avg_up = change_up.rolling(14).mean()
    avg_down = change_down.rolling(14).mean().abs()

    rsi = 100 * avg_up / (avg_up + avg_down)
    return rsi


def computeATR(stock_data: pd.DataFrame, period: int = 14):
    """Compute the Average True Range (ATR) of a stock."""
    high_low = stock_data['High'] - stock_data['Low']
    high_close = (stock_data['High'] - stock_data['Close'].shift()).abs()
    low_close = (stock_data['Low'] - stock_data['Close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def computeWMA(data, period):
    weights = np.arange(period, 0, -1)
    return data.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)


def computeMACD(data, short_period, long_period):
    short_EMA = data.ewm(span=short_period, adjust=False).mean()
    long_EMA = data.ewm(span=long_period, adjust=False).mean()
    MACD_line = short_EMA - long_EMA
    signal_line = MACD_line.ewm(span=9, adjust=False).mean()
    return MACD_line, signal_line


def computeDMI_ADX(high, low, close, lookback):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1/lookback).mean()

    return plus_di, minus_di, adx_smooth


def plotPriceHistory(pieClass, pieData):
    """Plot the history of the pie for each stock/ETF"""

    total_rows = len(pieData)
    for iter in range(0, total_rows, 2):
        current_rows = min(2, total_rows-iter)
        fig, axes = plt.subplots(nrows=current_rows, ncols=1, sharex=True)

        for index in range(current_rows):
            # Check if axes is not an array for pies with a only one remaining
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            stock = pieData[index+iter]
            axes[index].plot(stock['Close'].index, stock['Close'],
                             label=pieClass.holdings[index+iter])

            # Plot the Exponential Moving Avg and the Simple Moving Avg
            axes[index].plot(stock['SMA_50'].index,
                             stock['SMA_50'], label='SMA (50 days)', linestyle='--')
            axes[index].plot(stock['EMA_50'].index,
                             stock['EMA_50'], label='EMA (50 days)', linestyle='--')
            axes[index].plot(stock['WMA'].index,
                             stock['WMA'], label='WMA (14 days)', linestyle='--')

            axes[index].set_ylabel('Price/Indicator')
            axes[index].legend()
            axes[index].grid()

        # Create subplots for Close Price and RSI
        fig, axes_price_rsi = plt.subplots(
            nrows=current_rows, ncols=1, sharex=True)
        for index in range(current_rows):
            # Check if axes is not an array for pies with a only one remaining
            if not isinstance(axes_price_rsi, np.ndarray):
                axes_price_rsi = [axes_price_rsi]

            stock = pieData[index + iter]
            # Plot Close Price
            axes_price_rsi[index].plot(
                stock.index, stock['Close'], label='Close Price')

            # Plot RSI with Overbought and Oversold levels
            axes_rsi = axes_price_rsi[index].twinx()
            axes_rsi.plot(
                stock.index, stock['RSI'], label='RSI', color='orange')
            axes_rsi.axhline(y=70, color='red', linestyle='--',
                             label='Overbought (70)')
            axes_rsi.axhline(y=30, color='green',
                             linestyle='--', label='Oversold (30)')

            axes_price_rsi[index].set_ylabel('Price')
            axes_rsi.set_ylabel('RSI')
            axes_price_rsi[index].legend()
            axes_rsi.legend(loc='upper right')
            axes_price_rsi[index].grid()

        plt.suptitle("Stock/ETF Price History")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

        fig, axes_price_atr = plt.subplots(
            nrows=current_rows, ncols=1, sharex=True)
        for index in range(current_rows):
            # Check if axes is not an array for pies with a only one remaining
            if not isinstance(axes_price_atr, np.ndarray):
                axes_price_atr = [axes_price_atr]

            stock = pieData[index + iter]
            axes_price_atr[index].plot(
                stock.index, stock['ATR'], label='ATR (14 days)')
            axes_price_atr[index].legend()
            axes_price_atr[index].grid()
        plt.suptitle(f"Average True Range for Stocks/ETFs")
        plt.xlabel("Date")
        plt.xlabel("ATR")
        plt.tight_layout()
        plt.show()

        fig, axes_price_macd = plt.subplots(
            nrows=current_rows, ncols=1, sharex=True)
        for index in range(current_rows):
            # Check if axes is not an array for pies with a only one remaining
            if not isinstance(axes_price_macd, np.ndarray):
                axes_price_macd = [axes_price_macd]

            stock = pieData[index + iter]
            axes_price_macd[index].plot(
                stock.index, stock['MACD'], label='MACD (12-26 period EMA)')
            axes_price_macd[index].plot(
                stock.index, stock['MACD_Signal'], label='MACD Signal (12-26 period EMA)')
            axes_price_macd[index].grid()
            axes_price_macd[index].legend()
        plt.suptitle(f"Moving Average Convergence Divergence for Stocks/ETFs")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.tight_layout()
        plt.show()


def getPieData_Crypto(ok: bool = None):
    """True if you want to see the price plots with the indicators\n
    False if you don't want to plot the price history with indicators"""
    if ok is not None:
        crypto = Crypto()

        pieData_crypto = dataSetup(crypto.pieData)
        for index, name in enumerate(crypto.holdings):
            with open(f'./crypto/{name}.csv', 'w') as file:
                pieData_crypto[index].to_csv(file)
    if ok is True:
        plotPriceHistory(crypto, pieData_crypto)


def getPieData_ETFs(ok: bool = None):
    """True if you want to see the price plots with the indicators\n
    False if you don't want to plot the price history with indicators"""
    if ok is not None:
        etfs = ETFs()

        pieData_etfs = dataSetup(etfs.pieData)

        for index, name in enumerate(etfs.holdings):
            with open(f'./etfs/{name}.csv', 'w') as file:
                pieData_etfs[index].to_csv(file)
    if ok is True:
        plotPriceHistory(etfs, pieData_etfs)


def getPieData_EurozoneInvestments(ok: bool = None):
    """True if you want to plot the price history with the indicators\n
    False if you don't want to plot the price history with indicators"""
    if ok is not None:
        eurozone = EurozoneInvestments()

        pieData_eurozone = dataSetup(eurozone.pieData)
        for index, name in enumerate(eurozone.holdings):
            with open(f'./eurozone/{name}.csv', 'w') as file:
                pieData_eurozone[index].to_csv(file)
    if ok is True:
        plotPriceHistory(eurozone, pieData_eurozone)


def getPieData_OwnTheWorldIn50(ok: bool = None):
    """True if you want to see the price plots with the indicators\n
    False if you don't want to plot the price history with indicators"""
    if ok is not None and ok:
        otw = OwnTheWorldIn50()

        pieData_otw = dataSetup(otw.pieData)
        for index, name in enumerate(otw.holdings):
            with open(f'./otw/{name}.csv', 'w') as file:
                pieData_otw[index].to_csv(file)
    if ok:
        plotPriceHistory(otw, pieData_otw)


def main():
    while True:
        print(
            "[1] Eurozone Investments\n[2] Own the World in 50\n[3] World ETFs\n[4] Crypto")
        user_choice = int(
            input("What pie do you want to analyze?(Enter a number from above)\n"))
        price_history = input(
            "Do you want to plot the price history with indicators for this pie?[y/n]\n")
        if price_history == 'n' or price_history == 'N' or price_history == '':
            ok = False
        elif price_history == 'y' or price_history == 'Y':
            ok = True
        match user_choice:
            case 1:
                getPieData_EurozoneInvestments(ok)
                break
            case 2:
                getPieData_OwnTheWorldIn50(ok)
                break
            case 3:
                getPieData_ETFs(ok)
                break
            case 4:
                getPieData_Crypto(ok)
                break
            case _:
                print("Invalid choice. Try again...\n")
                continue


if __name__ == "__main__":
    main()
