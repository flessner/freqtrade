from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, IntParameter
from pandas import DataFrame
import numpy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Seb
# source: https://github.com/davidzr/freqtrade-strategies/blob/main/strategies/Seb/Seb.py

leverage = 1


class Seb(IStrategy):

    # Buy hyperspace params:
    buy_params = {
        "buy_fastd": 1,
        "buy_fishRsiNorma": 5,
        "buy_rsi": 26,
        "buy_volumeAVG": 150,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_fishRsiNorma": 30,
        "sell_minusDI": 4,
        "sell_rsi": 74,
        "sell_trigger": "rsi-macd-minusdi",
    }

    # ROI table:
    minimal_roi = {
        "0": 0.04 * leverage,
        "10": 0.025 * leverage,
        "25": 0.01 * leverage,
        "50": 0,
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.10

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # trailing stoploss
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = False

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    buy_volumeAVG = IntParameter(
        low=50,
        high=300,
        default=buy_params["buy_volumeAVG"],
        space="buy",
        optimize=True,
    )
    buy_rsi = IntParameter(
        low=1, high=100, default=buy_params["buy_rsi"], space="buy", optimize=True
    )
    buy_fastd = IntParameter(
        low=1, high=100, default=buy_params["buy_fastd"], space="buy", optimize=True
    )
    buy_fishRsiNorma = IntParameter(
        low=1, high=100, default=30, space=buy_params["buy_fishRsiNorma"], optimize=True
    )

    sell_rsi = IntParameter(
        low=1, high=100, default=sell_params["sell_rsi"], space="sell", optimize=True
    )
    sell_minusDI = IntParameter(
        low=1,
        high=100,
        default=sell_params["sell_minusDI"],
        space="sell",
        optimize=True,
    )
    sell_fishRsiNorma = IntParameter(
        low=1,
        high=100,
        default=sell_params["sell_fishRsiNorma"],
        space="sell",
        optimize=True,
    )
    sell_trigger = CategoricalParameter(
        ["rsi-macd-minusdi", "sar-fisherRsi"],
        default=sell_params["sell_trigger"],
        space="sell",
        optimize=True,
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe["ha_open"] = heikinashi["open"]
        dataframe["ha_close"] = heikinashi["close"]

        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["rsi"] = ta.RSI(dataframe)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_lowerband"] = bollinger["lowerband"]
        dataframe["bb_middleband"] = bollinger["middleband"]
        dataframe["bb_upperband"] = bollinger["upperband"]

        # Stoch
        stoch = ta.STOCH(dataframe)
        dataframe["slowk"] = stoch["slowk"]

        # Commodity Channel Index: values Oversold:<-100, Overbought:>100
        dataframe["cci"] = ta.CCI(dataframe)

        # Stoch
        stoch = ta.STOCHF(dataframe, 5)
        dataframe["fastd"] = stoch["fastd"]
        dataframe["fastk"] = stoch["fastk"]
        dataframe["fastk-previous"] = dataframe.fastk.shift(1)
        dataframe["fastd-previous"] = dataframe.fastd.shift(1)

        # Slow Stoch
        slowstoch = ta.STOCHF(dataframe, 50)
        dataframe["slowfastd"] = slowstoch["fastd"]
        dataframe["slowfastk"] = slowstoch["fastk"]
        dataframe["slowfastk-previous"] = dataframe.slowfastk.shift(1)
        dataframe["slowfastd-previous"] = dataframe.slowfastd.shift(1)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe["rsi"] - 50)
        dataframe["fisher_rsi"] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]

        # SAR Parabol
        dataframe["sar"] = ta.SAR(dataframe)

        # Hammer: values [0, 100]
        dataframe["CDLHAMMER"] = ta.CDLHAMMER(dataframe)

        # SMA - Simple Moving Average
        dataframe["sma"] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["ema20"], dataframe["ema50"])
                & (dataframe["ha_close"] > dataframe["ema20"])
                & (dataframe["ha_open"] < dataframe["ha_close"])  # green bar
            ),
            "buy",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["ema50"], dataframe["ema100"])
                & (dataframe["ha_close"] < dataframe["ema20"])
                & (dataframe["ha_open"] > dataframe["ha_close"])  # red bar
            ),
            "sell",
        ] = 1
        return dataframe
