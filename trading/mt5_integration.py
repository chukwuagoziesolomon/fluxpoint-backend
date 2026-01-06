import MetaTrader5 as mt5
from datetime import datetime, timedelta
from .models import Candle
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

# Timeframe mapping from string to MT5 constants
TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1,
}

TIMEFRAME_DELTA = {
    'M1': timedelta(minutes=1),
    'M5': timedelta(minutes=5),
    'M15': timedelta(minutes=15),
    'M30': timedelta(minutes=30),
    'H1': timedelta(hours=1),
    'H4': timedelta(hours=4),
    'D1': timedelta(days=1),
    'W1': timedelta(weeks=1),
    'MN1': timedelta(days=30),
}

class MT5DataIngestion:
    """
    Handles connection to MT5 terminal and data ingestion.
    """

    def __init__(self, login=None, password=None, server=None):
        """
        Initialize MT5 connection.

        :param login: MT5 account login
        :param password: MT5 account password
        :param server: MT5 server
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self):
        """
        Connect to MT5 terminal.

        :return: True if connected, False otherwise
        """
        if not mt5.initialize():
            logger.error("MT5 initialize failed")
            return False

        if self.login and self.password and self.server:
            if not mt5.login(self.login, self.password, self.server):
                logger.error("MT5 login failed")
                mt5.shutdown()
                return False

        self.connected = True
        logger.info("MT5 connected successfully")
        return True

    def disconnect(self):
        """
        Disconnect from MT5 terminal.
        """
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def fetch_historical_data(self, symbol, timeframe, start_date, end_date=None):
        """
        Fetch historical OHLCV data from MT5.

        :param symbol: Trading symbol, e.g., 'EURUSD'
        :param timeframe: Timeframe string, e.g., 'M15'
        :param start_date: Start date as datetime
        :param end_date: End date as datetime, defaults to now
        :return: List of rate data
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return []

        if end_date is None:
            end_date = datetime.now()

        mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
        if not mt5_timeframe:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return []

        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        if rates is None:
            logger.warning(f"Failed to fetch rates for {symbol} {timeframe} from {start_date} to {end_date}")
            return []

        logger.info(f"Fetched {len(rates)} rates for {symbol} {timeframe}")
        return rates

    def save_to_database(self, symbol, timeframe, rates):
        """
        Save fetched rates to Candle model.

        :param symbol: Trading symbol
        :param timeframe: Timeframe string
        :param rates: List of rate tuples from MT5
        """
        candles_to_create = []
        for rate in rates:
            timestamp = datetime.fromtimestamp(rate['time'])
            timestamp = timezone.make_aware(timestamp)

            if not Candle.objects.filter(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=timestamp
            ).exists():
                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    open_price=rate['open'],
                    high_price=rate['high'],
                    low_price=rate['low'],
                    close_price=rate['close'],
                    volume=rate['tick_volume']
                )
                candles_to_create.append(candle)

        if candles_to_create:
            Candle.objects.bulk_create(candles_to_create)
            logger.info(f"Saved {len(candles_to_create)} new candles for {symbol} {timeframe}")
        else:
            logger.debug(f"No new candles to save for {symbol} {timeframe}")

    def get_latest_timestamp(self, symbol, timeframe):
        try:
            latest = Candle.objects.filter(symbol=symbol, timeframe=timeframe).order_by('-timestamp').first()
            return latest.timestamp if latest else None
        except Exception as e:
            logger.error(f"Error getting latest timestamp for {symbol} {timeframe}: {e}")
            return None

    def ingest_data(self, symbols, timeframes, mode='incremental', days_back=30):
        """
        Ingest data for multiple symbols and timeframes.

        :param symbols: List of symbols, e.g., ['EURUSD', 'GBPUSD']
        :param timeframes: List of timeframes, e.g., ['M15', 'H1']
        :param mode: 'incremental' or 'full'
        :param days_back: Number of days back to fetch for full mode or initial
        """
        for symbol in symbols:
            for timeframe in timeframes:
                if mode == 'incremental':
                    latest_ts = self.get_latest_timestamp(symbol, timeframe)
                    if latest_ts:
                        start_date = latest_ts + TIMEFRAME_DELTA[timeframe]
                    else:
                        start_date = datetime.now() - timedelta(days=days_back)
                else:
                    start_date = datetime.now() - timedelta(days=days_back)

                rates = self.fetch_historical_data(symbol, timeframe, start_date)
                if rates:
                    self.save_to_database(symbol, timeframe, rates)