import yfinance as yf


def getPieData(holdings: list = None):
    stock_data = list()
    for stock in holdings:
        try:
            response = yf.download(stock)
            if response.empty:
                continue
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
        # IMAE.AS not added due to API having low data -> switched with XSX6.DE
        self.holdings = ['XSX6.DE', 'QDVE.DE', 'EUNL.DE',
                         'IS3N.DE', '4GLD.DE', 'EUNA.DE', 'XHYA.DE']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class EurozoneInvestments:
    def __init__(self):
        # 'DOU.DE', commenting out Douglas due to the low history
        self.holdings = ['AAPL', 'MSFT', 'GOOGL',
                         'NVDA', 'META', 'AMZN', 'AI.MC', 'AMD', 'INTC', 'V', 'MA', 'PAY.L', 'ADBE', 'NFLX', 'QCOM', 'STMPA.PA', 'IFX.DE', 'SIE.DE', 'SHL.DE', 'ENR.DE', 'ELUXBS.XD', 'ERICBS.XD', 'VAR1.DE', 'SU.PA', 'KNEBV.HE', 'GEBN.SW', 'SMSN.IL', 'PHIA.AS', 'HUSQBS.XD', 'ASML.AS', 'NOKIA.HE', 'HON', 'OMV.DE', 'ENI.MI', 'IPCOS.XD', 'SHELL.AS', 'ENEL.MI', 'ENGI.PA', 'VIE.PA', 'EOAN.DE', 'MBG.DE', 'DTG.DE', 'BMW.DE', 'VOW.DE', 'PAH3.DE', 'F', 'RNO.PA', 'VOLVBS.XD', 'TSLA', 'RACE.MI', 'AML.L', 'CON.DE', 'PIRC.MI', 'BRE.MI', 'AIR.PA', 'BA', 'WIZZ.L', 'AF.PA', 'LHA.DE', 'EZJ.L', 'SAF.PA', 'INGA.AS', 'DBK.DE', 'GLE.PA', 'UCG.MI', 'LSEG.L', 'BARC.L', 'GS', 'MS', 'JPM', 'C', 'UBSG.SW', 'G.MI', 'NN.AS', 'EDEN.PA', 'PLX.PA', 'SW.PA', 'SGO.PA', 'ZAL.DE', 'CA.PA', 'B4B.DE', 'CPR.MI', 'NESN.SW', 'BN.PA', 'PG', 'UNA.AS', 'MCD', 'KO', 'PEP', 'HEIA.AS', 'CARLBC.XD', 'PM', 'BATS.L', 'RMS.PA', 'CDI.PA', 'BRBY.L', 'MONC.MI', 'CPRI', 'MC.PA', 'BOSS.DE', 'OR.PA', 'PNDORA.CO', 'HMBS.XD', 'EL.PA', 'AFX.DE', 'GSK.L', 'ROG.SW', 'NOVOBC.XD', 'NOVN.SW', 'SAN.PA', 'BAYN.DE', 'PFE', 'AZN', 'MCOVBS.XD', 'SAP.DE', 'ABBN.SW', 'UBER', 'ABNB', 'BKNG', 'CAP.PA', 'VOD.L', 'ORA.PA', 'DTE.DE', 'SCMN.SW', 'ADS.DE', 'PUM.DE', 'NKE']

        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class Crypto:
    def __init__(self):
        self.holdings = ['BTC-EUR', 'ETH-EUR', 'ADA-EUR']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class Commodities:
    def __init__(self):
        self.holdings = [
            'CL=F', 'BZ=F', 'NG=F',   # Energy
            'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',   # Metals
            'ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'CC=F', 'SB=F', 'CT=F',   # Agricultural
            'LE=F', 'HE=F',   # Livestock
            'LBS=F', 'OJ=F'   # Other
        ]
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class DollarIndex:
    def __init__(self):
        self.holdings = ['DX-Y.NYB']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)


class StockIndices:
    def __init__(self):
        self.holdings = ['^FCHI', '^FTSE', '^GDAXI',
                         '^DJI', '^HSI', '^IXIC', '^RUT', '^NYA']
        print(
            f"<INFO> Fetching pie holdings historical data for {self.holdings}")
        self.pieData = getPieData(holdings=self.holdings)
