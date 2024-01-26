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
        self.holdings = ['BP', 'MSFT', 'AAPL', 'O', 'PEP', 'HON', 'V', 'MA', 'PG', 'AVGO', 'ULVR.L', 'AMT', 'CMCSA', 'JPM', 'ENB', 'NG', 'CL', 'ED', 'GPC', 'ITW', 'APD', 'LEG', 'MAIN', 'BLK', 'CSCO',
                         'D', 'CVX', 'UPS', 'STAG', 'HSBA.L', 'SMDS.L', 'HPE', 'RIO', 'ADM', 'BHP', 'DOW', 'AWR', 'KR', 'SGRO', 'YUM', 'REL', 'TSCO', 'KMI', 'FAST', 'PLD', 'WHR', 'CMI', 'WSM', 'SHEL', 'BEPC']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class ETFs:
    def __init__(self):
        self.holdings = ['QDVE.DE', 'SXRV.DE', 'SXR8.DE', 'EUNL.DE', 'VWCE.DE']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class EurozoneInvestments:
    def __init__(self):
        self.holdings = ['APC.DE', 'MSF.DE', 'ABEA.DE', 'NVD.DE', 'ABEC.DE', 'FB2A.DE', 'MA', 'ADBE', 'NFC.DE', 'EBS.VI', 'AMD', 'QCI.DE', '3V64.DE',
                         'INGA.AS', 'INTC', 'RMS.PA', 'OMV.VI', 'MBG.DE', 'BMW.DE', 'DBK.DE', 'AIR.DE', 'OR.PA', 'RBI.VI', 'BCO.DE', 'ADS.DE', 'ORA.PA', 'GLE.PA']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class Crypto:
    def __init__(self):
        self.holdings = ['BTC-EUR', 'ETH-EUR', 'ADA-EUR']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)
