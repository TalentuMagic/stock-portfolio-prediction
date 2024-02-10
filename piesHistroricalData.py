import yfinance as yf


def getPieData(holdings: list = None):
    stock_data = list()
    for stock in holdings:
        try:
            response = yf.download(stock)
            stock_data.append(response)
        except Exception as e:
            print(f"<ERROR> Failed fetching data for {stock}. {e}")
            continue

    return stock_data


class OwnTheWorldIn50:
    def __init__(self):
        self.holdings = ['CMCSA', 'MAIN', 'PLD', 'BP', 'MSFT', 'AAPL', 'O', 'PEP', 'HON', 'V', 'MA', 'PG', 'AVGO', 'ULVR.L', 'AMT', 'JPM', 'ENB', 'NG.L', 'CL', 'ED', 'GPC', 'ITW', 'APD', 'BLK', 'CSCO',
                         'CVX', 'UPS', 'STAG', 'HSBA.L', 'SMDS.L', 'HPE', 'RIO', 'ADM', 'BHP', 'DOW', 'KR', 'SGRO.L', 'YUM', 'REL', 'TSCO', 'KMI', 'FAST', 'WHR', 'CMI', 'WSM', 'SHEL.L', 'BEPC', 'LEG', 'D', 'AWR']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class ETFs:
    def __init__(self):
        self.holdings = ['QDVE.DE', 'SXRV.DE',
                         'SXR8.DE', 'EUNL.DE', 'VWCE.DE', '2B76.DE', 'XDWH.DE', 'XDWT.DE']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class EurozoneInvestments:
    def __init__(self):
        self.holdings = ['AAPL', 'MSFT', 'GOOGL',
                         'NVDA', 'META', 'MA', 'ADBE', 'NFLX', 'AMD', 'V', 'QCOM', 'INGA.AS', 'INTC', 'RMS.PA', 'OMV.DE', 'MBG.DE', 'BMW.DE', 'DBK.DE', 'AIR.PA', 'OR.PA', 'BA', 'F', 'ADS.DE', 'VOD', 'ORA.PA', 'DTE.DE', 'GLE.PA', 'PUM.DE', 'NKE', 'BAYN.DE', 'PM', 'BTI', 'SIE.DE', 'SHL.DE', 'ENR.DE', 'HON', 'ENEL.MI', 'ENGI.PA', 'ZAL.DE', 'CA.PA', 'B4B.DE', 'UNA.AS', 'PG', 'MCD', 'KO', 'PEP', 'HEIA.AS', 'SAP.DE', 'ABBN.SW', 'UBER', 'ABNB', 'BKNG', 'CAP.PA']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class Crypto:
    def __init__(self):
        self.holdings = ['BTC-EUR', 'ETH-EUR', 'ADA-EUR']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)
