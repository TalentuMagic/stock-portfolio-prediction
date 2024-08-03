import yfinance as yf
from analysePies import dataSetup


def getPieData(holdings: list = None):
    stock_data = list()
    for stock in holdings:
        try:
            response = yf.download(stock)
            if response.empty is True:
                print(f"[ERROR] {response}")
                return None
            stock_data.append(response)
        except Exception as e:
            print(f"<ERROR> Failed fetching data for {stock}. {e}")
            continue

    return stock_data


if __name__ == "__main__":

    ticker = str(input("Write the ticker of the stock/ETF:\n"))
    price_history = dataSetup(getPieData([ticker]))
    if price_history is None:
        raise SystemError("Failed to fetch data")
    with open(f'./extra/{ticker}.csv', 'w') as file:
        price_history[0].to_csv(file)
