from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame
import pandas
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter
)
from warnings import simplefilter

# Elliot8
# source: -

# ignore performance warning
simplefilter(action="ignore", category=pandas.errors.PerformanceWarning)


class E8(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_buy": 14,
        "ewo_high": 2.327,
        "ewo_low": -19.988,
        "low_offset": 0.975,
        "rsi_buy": 69,
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 24,
        "high_offset": 0.991,
        "high_offset_2": 0.997,
    }

    # ROI table:
    minimal_roi = {"0": 0.059, "10": 0.037, "41": 0.012, "114": 0}

    # Stoploss:
    stoploss = -0.03

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params["base_nb_candles_buy"], space="buy", optimize=True
    )
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params["base_nb_candles_sell"], space="sell", optimize=True
    )
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params["low_offset"], space="buy", optimize=True
    )
    high_offset = DecimalParameter(
        0.95, 1.1, default=sell_params["high_offset"], space="sell", optimize=True
    )
    high_offset_2 = DecimalParameter(
        0.99, 1.5, default=sell_params["high_offset_2"], space="sell", optimize=True
    )

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(
        -20.0, -8.0, default=buy_params["ewo_low"], space="buy", optimize=True
    )
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params["ewo_high"], space="buy", optimize=True
    )
    rsi_buy = IntParameter(
        30, 70, default=buy_params["rsi_buy"], space="buy", optimize=True
    )

    # Trailing stop:
    trailing_stop = False
    # trailing_stop_positive = 0.001
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Optimal timeframe for the strategy
    timeframe = "5m"
    inf_1h = "1h"

    process_only_new_candles = True
    startup_candle_count = 400

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate all ma_buy values
        for val in self.base_nb_candles_buy.range:
            dataframe[f"ma_buy_{val}"] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f"ma_sell_{val}"] = ta.EMA(dataframe, timeperiod=val)

        dataframe["hma_50"] = qtpylib.hull_moving_average(dataframe["close"], window=50)

        dataframe["sma_9"] = ta.SMA(dataframe, timeperiod=9)
        # Elliot
        dataframe["EWO"] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe["rsi_slow"] = ta.RSI(dataframe, timeperiod=20)

        return dataframe.copy()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe["rsi_fast"] < 35)
                & (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                        * self.low_offset.value
                    )
                )
                & (dataframe["EWO"] > self.ewo_high.value)
                & (dataframe["rsi"] < self.rsi_buy.value)
                & (dataframe["volume"] > 0)
                & (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset.value
                    )
                )
            )
        )

        conditions.append(
            (
                (dataframe["rsi_fast"] < 35)
                & (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_buy_{self.base_nb_candles_buy.value}"]
                        * self.low_offset.value
                    )
                )
                & (dataframe["EWO"] < self.ewo_low.value)
                & (dataframe["volume"] > 0)
                & (
                    dataframe["close"]
                    < (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset.value
                    )
                )
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "buy"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                (dataframe["close"] > dataframe["hma_50"])
                & (
                    dataframe["close"]
                    > (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset_2.value
                    )
                )
                & (dataframe["rsi"] > 50)
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
            )
            | (
                (dataframe["close"] < dataframe["hma_50"])
                & (
                    dataframe["close"]
                    > (
                        dataframe[f"ma_sell_{self.base_nb_candles_sell.value}"]
                        * self.high_offset.value
                    )
                )
                & (dataframe["volume"] > 0)
                & (dataframe["rsi_fast"] > dataframe["rsi_slow"])
            )
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "sell"] = 1

        return dataframe


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df["close"] * 100
    return emadif
