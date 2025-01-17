#!/usr/bin/env python
# coding: utf-8

# THIS SCRIPT IS PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SCRIPT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# DISCLAIMER FOR FINANCIAL MARKETS:
# This script is distributed for educational purposes only and does not constitute financial advice.
# Trading cryptocurrencies is highly speculative and involves substantial risk of loss.
# The author of this script takes no responsibility for any financial losses resulting from its use.
# Users assume full responsibility for their trading decisions and are strongly encouraged
# to consult a licensed financial advisor before engaging in cryptocurrency trading.
#
# DONATIONS:
# Bitcoin (BTC): bc1qt63aq5cpumgmv79ssy3ufy6qmawff5ulmx48cm
# Monero (XMR): 86AseotWnLaYoiMGgK79QiC9wRXfCvmSyLBHaa5Z66xWQL3sSruXjGfbLGA3VmJcVWKpmWgUApcNGBUjuCPDLJtNDeewEus
# Litecoin (LTC): MKd5bRmRQ8NxAbuP6rkWvF15BFYpG95KFs


import numpy as np
import pandas as pd
import argparse
import pickle
import warnings
from datetime import datetime
import logging
import _thread
import configparser
import websockets
import asyncio
import json
import signal
import threading
from decimal import *
from time import sleep
from binance.um_futures import UMFutures
from binance.lib.utils import config_logging
from binance.error import ClientError

# Constants
MODEL_FILE = 'model.pkl'
SYMBOL = 'RSRUSDT'
INTERVAL = '4h'
PRECISION = 0
CLOSE_ALL_ON_EXIT = True

config_logging(logging, logging.ERROR)

class NNTrader():
    def __init__(self, symbol, bar, usd, precision, algorithm):
        self.model = algorithm['model']
        self.features = algorithm['features']
        self.lags = algorithm['lags']

        self.symbol = symbol
        self.symbol_configuration = None
        self.bar = bar

        self.usd = usd                     # The number of usd traded
        self.prev_price = None
        self.precision = precision         # Precision to calculate units
        self.position = 0               # The initial, neutral position

        self.data = pd.DataFrame()
        self.cols = []
        self.raw_data = pd.DataFrame()

        self.client = BinanceFuturesApi('config.cfg', self.symbol, self.bar)
        self.symbol_configuration = self.client.symbol_configuration(self.symbol)
        print(f"Leverage: {self.symbol_configuration['leverage']}")

    def fill_data_with_history(self):
        history_df = self.client.get_history(limit=100)
        self.data = pd.concat([self.data, history_df])
        self.prepare_features()

    def prepare_features(self):
        epsilon = 1e-6

        small_period = 3
        medium_period = 9
        large_period = 20

        # Calculates the log returns from the closing prices
        self.data['return'] = np.log(self.data['c'] / self.data['c'].shift(1))

        self.data['return_s'] = np.log(self.data['c'] / self.data['c'].shift(small_period))
        self.data['return_m'] = np.log(self.data['c'] / self.data['c'].shift(medium_period))
        self.data['return_l'] = np.log(self.data['c'] / self.data['c'].shift(large_period))

        self.data['v'] = self.data['v'].replace(0, epsilon)
        self.data['volume_return'] = np.log(self.data['v'] / (self.data['v'].shift(1)))

        self.data['candle_body'] = self.data['c'] - self.data['o']

        self.data['upper_shadow'] = self.data.apply(lambda row: row['h'] - max(row['c'], row['o']), axis=1)
        self.data['lower_shadow'] = self.data.apply(lambda row: min(row['c'], row['o']) - row['l'], axis=1)

        self.data['candle_range'] = self.data['h'] - self.data['l']

        self.data['candle_body_ratio'] = self.data['candle_body'] / (self.data['candle_range'] + epsilon)
        self.data['upper_shadow_ratio'] = self.data['upper_shadow'] / (self.data['candle_range'] + epsilon)
        self.data['lower_shadow_ratio'] = self.data['lower_shadow'] / (self.data['candle_range'] + epsilon)

        self.data['mean_s'] = self.data['c'].rolling(small_period, min_periods=1).mean()
        self.data['mean_m'] = self.data['c'].rolling(medium_period, min_periods=1).mean()
        self.data['mean_l'] = self.data['c'].rolling(large_period, min_periods=1).mean()
        self.data['volatility_s'] = self.data['c'].rolling(small_period, min_periods=1).std()
        self.data['volatility_m'] = self.data['c'].rolling(medium_period, min_periods=1).std()
        self.data['volatility_l'] = self.data['c'].rolling(large_period, min_periods=1).std()

        self.data['deviation_from_mean_s'] = self.data['c'] - self.data['mean_s']
        self.data['deviation_from_mean_m'] = self.data['c'] - self.data['mean_m']
        self.data['deviation_from_mean_l'] = self.data['c'] - self.data['mean_l']

        self.data['deviation_ratio_s'] = self.data['deviation_from_mean_s'] / self.data['mean_s']
        self.data['deviation_ratio_m'] = self.data['deviation_from_mean_m'] / self.data['mean_m']
        self.data['deviation_ratio_l'] = self.data['deviation_from_mean_l'] / self.data['mean_l']

        self.data['ATR_s'] = self.calculate_atr(self.data, small_period)
        self.data['ATR_m'] = self.calculate_atr(self.data, medium_period)
        self.data['ATR_l'] = self.calculate_atr(self.data, large_period)

        #self.data.dropna(inplace=True)

        self.cols = []
        for f in self.features:
            for lag in range(0, self.lags):
                col = f'{f}_lag_{lag}'
                self.data[col] = self.data[f].shift(lag)
                self.cols.append(col)


    def calculate_atr(self, data, window=14):
        high_low = data['h'] - data['l']
        high_close = np.abs(data['h'] - data['c'].shift())
        low_close = np.abs(data['l'] - data['c'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def on_success(self, data):
        ''' Method called when new data is retrieved. '''

	#     "o": "0.0010",      // Open price
	#     "c": "0.0020",      // Close price
	#     "h": "0.0025",      // High price
	#     "l": "0.0015",      // Low price
	#     "v": "1000",        // Base asset volume
	#     "n": 100,           // Number of trades
	#     "x": false,         // Is this kline closed?
	#     "q": "1.0000",      // Quote asset volume
	#     "V": "500",         // Taker buy base asset volume
	#     "Q": "0.500",       // Taker buy quote asset volume

        if( data.iloc[-1]['x'] == True ):

            self.data = pd.concat([self.data, data[['o','c','h','l','v','V','n','q','Q']] ])

            self.prepare_features()

            features = self.data[self.cols].tail(1) # self.data[self.cols].iloc[-1]

            signal = self.model.predict(features)[0]

            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            price_close = self.data["c"].iloc[-1]
            growth_decay = 0.9

            if self.prev_price is not None:
                price_rel = price_close / self.prev_price


            if self.position in [0, -1] and signal == 1:
                if self.prev_price is not None:
                    if self.position == -1:
                        coef = self.symbol_configuration['leverage'] + 1 - self.symbol_configuration['leverage'] * price_rel

                        if coef > 1:
                            coef = (coef - 1) * growth_decay + 1

                        self.usd = round( coef * self.usd, 2)

                units = round(self.usd/price_close, self.precision)

                print(f'{dt_string} *** GOING LONG *** price: {price_close} usd: {self.usd} units: {units}')
                self.client.close_all_positions()
                self.client.open_position('BUY', 'LONG', units)
                self.position = 1
                self.prev_price = price_close
            elif self.position in [0, 1] and signal == -1:
                if self.prev_price is not None:
                    if self.position == 1:
                        coef = self.symbol_configuration['leverage'] * price_rel - self.symbol_configuration['leverage'] + 1

                        if coef > 1:
                            coef = (coef - 1) * growth_decay + 1

                        self.usd = round( coef * self.usd, 2)

                units = round(self.usd/price_close, self.precision)

                print(f'{dt_string} *** GOING SHORT *** price: {price_close} usd: {self.usd} units: {units}')
                self.client.close_all_positions()
                self.client.open_position('SELL', 'SHORT', units)
                self.position = -1
                self.prev_price = price_close
            else:
                print(f'{dt_string} price: {price_close}')


class BinanceFuturesApi(object):
    ''' BinanceApi is a Python wrapper class for the Binance Api. '''

    def __init__(self, conf_file, symbol, bar):
        ''' Init function is expecting a configuration file with
        the following content:

        [BINANCE]
        mode=live
        key=
        secret=

        Parameters
        ==========
        conf_file: string
            path to and filename of the configuration file,
            e.g. '/home/me/config.cfg'
        '''
        self.config = configparser.ConfigParser()
        self.config.read(conf_file)
        self.api_key = self.config['BINANCE']['key']
        self.api_secret = self.config['BINANCE']['secret']

        if(self.config['BINANCE']['mode']=='test'):
            self.client = UMFutures(key=self.api_key, secret=self.api_secret, base_url='https://testnet.binancefuture.com')
        else:
            self.client = UMFutures(key=self.api_key, secret=self.api_secret)

        self.symbol = symbol
        self.bar = bar                 # The bar length on which the algorithm is implemented



    def close_all_positions(self):
        positions = self.get_positions_info()

        for position in positions:

            if float(position['positionAmt']) == 0.0:
                continue

            if position['positionSide'] == 'LONG':
                self.open_position( 'SELL', 'LONG', abs(float(position['positionAmt'])) )

            if position['positionSide'] == 'SHORT':
                self.open_position('BUY', 'SHORT', abs(float(position['positionAmt'])) )


    def open_position(self, side, positionSide, quantity, **kwargs):
        kwargs['quantity'] = quantity
        kwargs['positionSide'] = positionSide
        self.create_order('MARKET', side, **kwargs)

    def create_order(self, type, side, **kwargs):
        '''
        https://binance-docs.github.io/apidocs/futures/en/#new-order-trade
        '''

        params = {
                'symbol': self.symbol,
                'side': side,
                'type': type,                   # 'LIMIT' 'MARKET' STOP/TAKE_PROFIT STOP_MARKET/TAKE_PROFIT_MARKET
        }

        if 'positionSide' in kwargs:
            params['positionSide'] = kwargs['positionSide']

        if 'quantity' in kwargs:
            params['quantity'] = kwargs['quantity']

        if type == 'LIMIT':
            params['timeinforce'] = 'GTC'

        if 'price' in kwargs:
            params['price'] = kwargs['price']

        if 'closePosition' in kwargs and kwargs['closePosition'] == True:
            params['closePosition'] = True

        try:
            response = self.client.new_order(**params)
            logging.info(response)
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

    def symbol_configuration(self, symbol, **kwargs):
        '''
        https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/Symbol-Config
        '''
        kwargs['symbol'] = symbol
        response = self.client.symbol_configuration(**kwargs)
        return response[0] if response else None

    def get_exchange_info(self):
        '''
        https://binance-docs.github.io/apidocs/futures/en/#exchange-information
        '''
        return self.client.exchange_info()

    def get_account_positions_info(self):
        resp = self.get_account_info()
        return list(filter(lambda x: x['symbol'] == self.symbol, resp['positions']))

    def get_positions_info(self):
        '''
        :API doc: https://binance-docs.github.io/apidocs/futures/en/#position-information-v2-user_data
        '''
        resp = self.client.get_position_risk(symbol=self.symbol, recvWindow=10000)

        return resp

    def get_account_balance(self):
        return self.client.balance()

    def get_account_info(self):
        '''
        :API doc: https://binance-docs.github.io/apidocs/futures/en/#account-information-v2-user_data
        '''
        return self.client.account()

    def get_server_time(self):
        return self.client.time()


    # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
    def get_history(self, **kwargs):
        """
        |
        | **Kline/Candlestick Data**
        | *Kline/candlestick bars for a symbol. Klines are uniquely identified by their open time.*

        :API endpoint: ``GET /fapi/v1/klines``
        :API doc: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data

        :parameter symbol: string; the trading symbol.
        :parameter interval: string; the interval of kline, e.g 1m, 5m, 1h, 1d, etc. (see more in https://binance-docs.github.io/apidocs/futures/en/#public-endpoints-info)
        :parameter limit: optional int; limit the results. Default 500, max 1000.
        :parameter startTime: optional int
        :parameter endTime: optional int
        |
        """
        klines = pd.DataFrame(self.client.klines(self.symbol, self.bar, **kwargs), 
                    columns=['t', 'o', 'h', 'l', 'c', 'v', 'T', 'q', 'n', 'V', 'Q', 'B'])
 
        klines = klines.set_index('t')

        klines.index = pd.to_datetime(klines.index, unit='ms').tz_localize(None)

        klines = klines.drop(columns=['T', 'B'])

        # Remove the last row because it represents an incomplete (not closed) kline
        klines = klines[:-1]

        klines = klines.astype({'o':float, 'h':float, 'l':float, 'c':float, 'v':float, 'q':float, 'n':int, 'V':float, 'Q':float})

        return klines

    def on_success(self, data):
        print(data)

    def stream_data(self, callback=None):
        ''' 

        https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams

        Starts a real-time data stream.

        Parameters
        ==========
        symbol: string; the trading symbol
        '''

        uri_symbol_param = f'{self.symbol.lower()}@kline_{self.bar}'

        print("START STREAM: ", uri_symbol_param)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if str(e).startswith('There is no current event loop in thread'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise

        loop.run_until_complete(self._stream(uri_symbol_param, callback))
        loop.close()

    def stream_data_failsafe(self, callback=None):
        signal.signal(signal.SIGTERM, service_shutdown)
        signal.signal(signal.SIGINT, service_shutdown)
        signal.signal(signal.SIGSEGV, service_shutdown)
        try:
            price_stream_thread = Job(self._stream_data_failsafe_thread,
                                      [callback])
            price_stream_thread.start()

            # https://stackoverflow.com/questions/65467329/server-in-a-thread-python3-9-0aiohttp-runtimeerror-cant-register-atexit-a/68187929#68187929
            price_stream_thread.join()
            return price_stream_thread
        except ServiceExit as e:
            print('Handling exception')
            price_stream_thread.shutdown_flag.set()
            price_stream_thread.join()

    def _stream_data_failsafe_thread(self, args):
        try:
            print("Starting price streaming")
         
            self.stream_data(callback=args[0])
        except Exception as e:
            import sys
            import traceback
            print(traceback.format_exc())
            print('Sleeping..')
            sleep(3)
            return

    async def _stream(self, uri_symbol_param, callback=None):
        '''
        # https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
        '''
        url_raw = f"wss://fstream.binance.com/ws/{uri_symbol_param}"
        # url_combined = "wss://stream.binance.com/stream?streams=btcusdt@kline_1m/ltcusdt@kline_1m/batusdt@kline_1m"

        async with websockets.connect(url_raw) as client:
            while not threading.current_thread().shutdown_flag.is_set():
                raw_str = await client.recv()
                raw = json.loads(raw_str) # <class 'dict'>
                data = raw

                # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-streams
                # {
                #   "e": "kline",     // Event type
                #   "E": 1638747660000,   // Event time
                #   "s": "BTCUSDT",    // Symbol
                #   "k": {
                #     "t": 1638747660000, // Kline start time
                #     "T": 1638747719999, // Kline close time
                #     "s": "BTCUSDT",  // Symbol
                #     "i": "1m",      // Interval
                #     "f": 100,       // First trade ID
                #     "L": 200,       // Last trade ID
                #     "o": "0.0010",  // Open price
                #     "c": "0.0020",  // Close price
                #     "h": "0.0025",  // High price
                #     "l": "0.0015",  // Low price
                #     "v": "1000",    // Base asset volume
                #     "n": 100,       // Number of trades
                #     "x": false,     // Is this kline closed?
                #     "q": "1.0000",  // Quote asset volume
                #     "V": "500",     // Taker buy base asset volume
                #     "Q": "0.500",   // Taker buy quote asset volume
                #     "B": "123456"   // Ignore
                #   }
                # }
                df = pd.DataFrame({'o': float(data['k']['o']),
                           'c': float(data['k']['c']),
                           'h': float(data['k']['h']),
                           'l': float(data['k']['l']),
                           'v': float(data['k']['v']),
                           'n': int(data['k']['n']),
                           'x': bool(data['k']['x']),
                           'q': float(data['k']['q']),
                           'V': float(data['k']['V']),
                           'Q': float(data['k']['Q']),
                          }, index=[pd.Timestamp(data['k']['t'], unit='ms').tz_localize(None)])

                if callback is not None:
                    callback(df)
                else:
                    self.on_success(df)


def service_shutdown(signum, frame):
    print('exiting ...')
    raise ServiceExit

class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """

    def __init__(self, message=None):
        self.message = message

    def __repr__(self):
        return repr(self.message)




class Job(threading.Thread):
    def __init__(self, job_callable, args=None):
        threading.Thread.__init__(self)
        self.callable = job_callable
        self.args = args

        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()
        self.job = None
        self.exception = None

    def run(self):
        print('Thread #%s started' % self.ident)
        try:
            self.job = self.callable
            while not self.shutdown_flag.is_set():
                print("Starting job loop...")
                if self.args is None:
                    self.job()
                else:
                    self.job(self.args)
        except Exception as e:
            import sys
            import traceback
            print(traceback.format_exc())
            self.exception = e
            _thread.interrupt_main()



if __name__ == '__main__':

    # Example to start: python trade.py 23

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    parser = argparse.ArgumentParser(description='Trade on Binance Futures')
    parser.add_argument('usd', type=float, help='USD trade equivalent')

    args = parser.parse_args()

    with open(MODEL_FILE, 'rb') as file:
        algorithm = pickle.load(file)

    trader = NNTrader(SYMBOL.upper(), INTERVAL, args.usd, PRECISION, algorithm)

    trader.fill_data_with_history()

    trader.client.stream_data_failsafe(trader.on_success)

    if CLOSE_ALL_ON_EXIT:
        print('*** CLOSING OUT ***')
        trader.client.close_all_positions()
    else:
        print('*** CLOSE ALL POSITIONS MANUALLY ***')





